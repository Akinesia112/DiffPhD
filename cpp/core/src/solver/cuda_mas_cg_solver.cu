// cuda_mas_cg_solver.cu — GPU CG solver with MAS preconditioner.
//
// Implements CudaMasCgSolver declared in cuda_mas_cg_solver.h.
// All CUDA types (handles, device pointers) are hidden in the Impl struct.
//
// Design:
//  - SpMV:   cuSPARSE (double, CSR)
//  - Dots/axpy: cuBLAS (double), POINTER_MODE_HOST for simplicity
//  - MAS preconditioner: 3 custom kernels (float internally)
//  - CG loop entirely on GPU; host only syncs for convergence check every N iters
//
// MAS kernel reference: Wu et al., SIGGRAPH 2022, Sections 7.1-7.3.

#include "solver/cuda_mas_cg_solver.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// Error-checking macros
// ============================================================

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in " __FILE__ ":") \
            + std::to_string(__LINE__) + ": " + cudaGetErrorString(_e)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _e = (call); \
    if (_e != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS error ") \
            + std::to_string((int)_e) \
            + " in " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
} while(0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t _e = (call); \
    if (_e != CUSPARSE_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuSPARSE error ") \
            + std::to_string((int)_e) \
            + " in " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
} while(0)

// ============================================================
// CUDA Kernels
// ============================================================

// ------------------------------------
// Kernel 1a: BuildResidualHierarchy — level 0
// One thread per level-0 vertex.
// Reads double CG residual d_r_cg, writes float d_mappedR at level-0 positions.
// Atomically accumulates into level-1 via d_goingNext (32 fine nodes → 1 supernode).
// ------------------------------------
__global__ void BuildResidualHierarchyKernel(
    const double* __restrict__ d_r_cg,
    float* __restrict__        d_mappedR,
    const int* __restrict__    d_mapperSortedGetOriginal,
    const int* __restrict__    d_goingNext,
    int numVerts,
    int numLevel)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    int origIdx = d_mapperSortedGetOriginal[vid];
    float rx = (float)d_r_cg[3 * origIdx + 0];
    float ry = (float)d_r_cg[3 * origIdx + 1];
    float rz = (float)d_r_cg[3 * origIdx + 2];

    // Level-0: direct write (each vid is unique, no conflicts)
    d_mappedR[vid * 3 + 0] = rx;
    d_mappedR[vid * 3 + 1] = ry;
    d_mappedR[vid * 3 + 2] = rz;

    // Level-0 → level-1: atomic accumulate (multiple fine nodes share one coarse node)
    if (numLevel > 1) {
        int next = d_goingNext[vid];
        atomicAdd(&d_mappedR[next * 3 + 0], rx);
        atomicAdd(&d_mappedR[next * 3 + 1], ry);
        atomicAdd(&d_mappedR[next * 3 + 2], rz);
    }
}

// ------------------------------------
// Kernel 1b: PropagateResidualHierarchy — level 1 → 2 → ... → L-1
// One thread per level-1 vertex. Runs only when numLevel > 2.
// ------------------------------------
__global__ void PropagateResidualHierarchyKernel(
    float* __restrict__     d_mappedR,
    const int* __restrict__ d_goingNext,
    int bg,
    int ed,
    int numLevel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vid = bg + idx;
    if (vid >= ed) return;

    float rx = d_mappedR[vid * 3 + 0];
    float ry = d_mappedR[vid * 3 + 1];
    float rz = d_mappedR[vid * 3 + 2];

    int nx = vid;
    for (int lv = 2; lv < numLevel; lv++) {
        nx = d_goingNext[nx];
        atomicAdd(&d_mappedR[nx * 3 + 0], rx);
        atomicAdd(&d_mappedR[nx * 3 + 1], ry);
        atomicAdd(&d_mappedR[nx * 3 + 2], rz);
    }
}

// ------------------------------------
// Kernel 2: SchwarzLocalXSym — per-domain symmetric matvec
// One GPU thread per MAS domain block (32 nodes × 3 DOF = 96 elements).
// Directly ports the scalar fallback from SeSchwarzPreconditioner.cpp lines 1838–1883.
//
// invSymR compact format (triSz = 4656 floats per block):
//   [0..95]   diagonal D (96 elements)
//   [96..]    off-diagonal body: groups of 8, outer loop it=0..10,
//              inner loop scan=xBg+1..88, 8 floats per (it,scan)
//   [tail]    12 × 7 sparse tail entries
// ------------------------------------
__global__ void SchwarzLocalXSymKernel(
    const float* __restrict__ d_mappedR,
    float* __restrict__       d_mappedZ,
    const float* __restrict__ d_invSymR,
    int activeBlockNum)
{
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= activeBlockNum) return;

    constexpr int DOMAIN_SIZE = 32;
    constexpr int NDOF = 96;        // 32 nodes × 3
    constexpr int triSz = (1 + NDOF) * NDOF / 2 + 16 * 3;  // = 4656

    float cacheRhs[NDOF];
    float cacheOut[NDOF];

    // Load RHS from d_mappedR (stride-3 layout: [vid*3+0..2] = x,y,z)
    for (int laneIdx = 0; laneIdx < DOMAIN_SIZE; ++laneIdx) {
        int vid = laneIdx + blockId * DOMAIN_SIZE;
        cacheRhs[laneIdx * 3 + 0] = d_mappedR[vid * 3 + 0];
        cacheRhs[laneIdx * 3 + 1] = d_mappedR[vid * 3 + 1];
        cacheRhs[laneIdx * 3 + 2] = d_mappedR[vid * 3 + 2];
    }

    int off = blockId * triSz;

    // Phase 1: Diagonal — cacheOut[i] = invSymR[i] * cacheRhs[i]
    for (int i = 0; i < NDOF; i++)
        cacheOut[i] = d_invSymR[off + i] * cacheRhs[i];

    // Phase 2: Off-diagonal body (groups of 8, symmetric update)
    // Matching CPU scalar fallback: off += 8*12 = 96 after diagonal
    off += 8 * 12;
    for (int it = 0; it < 11; it++) {
        int xBg = it * 8;
        for (int scan = xBg + 1; scan <= (NDOF - 8); scan++) {
            for (int lane = 0; lane < 8; lane++) {
                float mtx = d_invSymR[off + lane];
                cacheOut[xBg  + lane] += mtx * cacheRhs[scan + lane];
                cacheOut[scan + lane] += mtx * cacheRhs[xBg  + lane];
            }
            off += 8;
        }
    }

    // Phase 3: Tail (12 × 7 sparse entries)
    for (int it = 0; it < 12; it++) {
        for (int lane = 0; lane < 7; lane++) {
            int xBg = it * 8 + lane;
            float sigularRhs    = cacheRhs[xBg];
            float singularResult = 0.0f;
            for (int h = (NDOF - 7 + lane); h < NDOF; h++) {
                float value = d_invSymR[off];
                off++;
                if (value != 0.0f) {
                    singularResult  += value * cacheRhs[h];
                    cacheOut[h]     += value * sigularRhs;
                }
            }
            cacheOut[xBg] += singularResult;
        }
    }

    // Store results to d_mappedZ
    for (int laneIdx = 0; laneIdx < DOMAIN_SIZE; ++laneIdx) {
        int vid = laneIdx + blockId * DOMAIN_SIZE;
        d_mappedZ[vid * 3 + 0] = cacheOut[laneIdx * 3 + 0];
        d_mappedZ[vid * 3 + 1] = cacheOut[laneIdx * 3 + 1];
        d_mappedZ[vid * 3 + 2] = cacheOut[laneIdx * 3 + 2];
    }
}

// ------------------------------------
// Kernel 3: CollectFinalZ
// One thread per level-0 vertex. Accumulates m_mappedZ from all levels and
// writes to CG z vector (float→double, with inverse Morton reordering).
// Ports CPU CollectFinalZ (SeSchwarzPreconditioner.cpp lines 1903-1926).
// ------------------------------------
__global__ void CollectFinalZKernel(
    double* __restrict__      d_z_cg,
    const float* __restrict__ d_mappedZ,
    const int* __restrict__   d_mapperSortedGetOriginal,
    const int* __restrict__   d_coarseTablesFlat,   // [numVerts * 4]
    int numVerts,
    int numLevel)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    int mappedIndex = d_mapperSortedGetOriginal[vid];

    float zx = d_mappedZ[vid * 3 + 0];
    float zy = d_mappedZ[vid * 3 + 1];
    float zz = d_mappedZ[vid * 3 + 2];

    // Accumulate from coarser levels (up to 3 additional levels, matching CPU's Min(numLevel, 4))
    int maxLevels = (numLevel < 4) ? numLevel : 4;
    for (int l = 1; l < maxLevels; l++) {
        int now = d_coarseTablesFlat[vid * 4 + (l - 1)];
        zx += d_mappedZ[now * 3 + 0];
        zy += d_mappedZ[now * 3 + 1];
        zz += d_mappedZ[now * 3 + 2];
    }

    // Convert float → double and write to CG output
    d_z_cg[3 * mappedIndex + 0] = (double)zx;
    d_z_cg[3 * mappedIndex + 1] = (double)zy;
    d_z_cg[3 * mappedIndex + 2] = (double)zz;
}

// ------------------------------------
// CG helper: p = z + beta * p
// ------------------------------------
__global__ void BetaUpdateKernel(double* __restrict__ p,
                                  const double* __restrict__ z,
                                  double beta, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = z[i] + beta * p[i];
}

// ============================================================
// Impl struct — holds all CUDA state
// ============================================================

struct CudaMasCgSolverImpl {
    cublasHandle_t   cublas_handle   = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;

    // CSR sparse matrix (Newton Hessian, double)
    int     num_rows = 0;
    int     nnz      = 0;
    int*    d_csr_row_ptr = nullptr;
    int*    d_csr_col_ind = nullptr;
    double* d_csr_values  = nullptr;

    // cuSPARSE generic SpMV descriptors
    cusparseSpMatDescr_t mat_A  = nullptr;
    cusparseDnVecDescr_t vec_p  = nullptr;
    cusparseDnVecDescr_t vec_Ap = nullptr;
    void*  spmv_buffer      = nullptr;
    size_t spmv_buffer_size = 0;

    // CG vectors (double, length = num_rows)
    double* d_x  = nullptr;  // solution
    double* d_r  = nullptr;  // residual
    double* d_p  = nullptr;  // search direction
    double* d_Ap = nullptr;  // A * p
    double* d_z  = nullptr;  // preconditioned residual

    // MAS preconditioner data (GPU, float)
    float* d_invSymR             = nullptr;
    int*   d_goingNext           = nullptr;
    int*   d_mapperSortedGetOrig = nullptr;
    int*   d_coarseTablesFlat    = nullptr;
    float* d_mappedR             = nullptr;
    float* d_mappedZ             = nullptr;

    // Scalar parameters
    int numVerts             = 0;
    int totalSz              = 0;
    int numLevel             = 0;
    int totalNumberClusters  = 0;

    // Level sizes kept on host for kernel launch parameters
    std::vector<int> h_levelSizeY;

    bool allocated = false;

    void Free() {
        if (!allocated) return;

        // cuSPARSE descriptors
        if (mat_A)  { cusparseDestroySpMat(mat_A); mat_A = nullptr; }
        if (vec_p)  { cusparseDestroyDnVec(vec_p); vec_p = nullptr; }
        if (vec_Ap) { cusparseDestroyDnVec(vec_Ap); vec_Ap = nullptr; }
        if (spmv_buffer) { cudaFree(spmv_buffer); spmv_buffer = nullptr; }

        // Sparse matrix
        if (d_csr_row_ptr) { cudaFree(d_csr_row_ptr); d_csr_row_ptr = nullptr; }
        if (d_csr_col_ind) { cudaFree(d_csr_col_ind); d_csr_col_ind = nullptr; }
        if (d_csr_values)  { cudaFree(d_csr_values);  d_csr_values  = nullptr; }

        // CG vectors
        if (d_x)  { cudaFree(d_x);  d_x  = nullptr; }
        if (d_r)  { cudaFree(d_r);  d_r  = nullptr; }
        if (d_p)  { cudaFree(d_p);  d_p  = nullptr; }
        if (d_Ap) { cudaFree(d_Ap); d_Ap = nullptr; }
        if (d_z)  { cudaFree(d_z);  d_z  = nullptr; }

        // MAS data
        if (d_invSymR)             { cudaFree(d_invSymR);             d_invSymR             = nullptr; }
        if (d_goingNext)           { cudaFree(d_goingNext);           d_goingNext           = nullptr; }
        if (d_mapperSortedGetOrig) { cudaFree(d_mapperSortedGetOrig); d_mapperSortedGetOrig = nullptr; }
        if (d_coarseTablesFlat)    { cudaFree(d_coarseTablesFlat);    d_coarseTablesFlat    = nullptr; }
        if (d_mappedR)             { cudaFree(d_mappedR);             d_mappedR             = nullptr; }
        if (d_mappedZ)             { cudaFree(d_mappedZ);             d_mappedZ             = nullptr; }

        // cuBLAS/cuSPARSE handles
        if (cublas_handle)   { cublasDestroy(cublas_handle);   cublas_handle   = nullptr; }
        if (cusparse_handle) { cusparseDestroy(cusparse_handle); cusparse_handle = nullptr; }

        allocated = false;
    }
};

// ============================================================
// CudaMasCgSolver — public interface
// ============================================================

CudaMasCgSolver::CudaMasCgSolver() : impl_(new CudaMasCgSolverImpl()) {}

CudaMasCgSolver::~CudaMasCgSolver() {
    impl_->Free();
    delete impl_;
}

void CudaMasCgSolver::Setup(
    int num_rows, int nnz,
    const int* csc_outer_ptr,
    const int* csc_inner_idx,
    const double* csc_values,
    const float* invSymR, int invSymR_size,
    const int* goingNext, int goingNext_size,
    const int* mapperSortedGetOriginal, int numVerts,
    const int* coarseTablesFlat,
    const int* levelSizeX,
    const int* levelSizeY,
    int numLevel,
    int totalSz,
    int totalNumberClusters)
{
    CudaMasCgSolverImpl* d = impl_;

    // If sizes changed, free and reallocate
    bool need_realloc = !d->allocated
        || d->num_rows != num_rows
        || d->nnz      != nnz
        || d->numVerts != numVerts
        || d->totalSz  != totalSz;

    if (need_realloc) {
        d->Free();

        // Create handles
        CUBLAS_CHECK(cublasCreate(&d->cublas_handle));
        CUSPARSE_CHECK(cusparseCreate(&d->cusparse_handle));
        CUBLAS_CHECK(cublasSetPointerMode(d->cublas_handle, CUBLAS_POINTER_MODE_HOST));

        // Allocate sparse matrix arrays
        CUDA_CHECK(cudaMalloc(&d->d_csr_row_ptr, (num_rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d->d_csr_col_ind, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d->d_csr_values,  nnz * sizeof(double)));

        // Allocate CG vectors
        CUDA_CHECK(cudaMalloc(&d->d_x,  num_rows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d->d_r,  num_rows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d->d_p,  num_rows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d->d_Ap, num_rows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d->d_z,  num_rows * sizeof(double)));

        // Allocate MAS data
        CUDA_CHECK(cudaMalloc(&d->d_invSymR,             invSymR_size  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d->d_goingNext,           goingNext_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d->d_mapperSortedGetOrig, numVerts * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d->d_coarseTablesFlat,    numVerts * 4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d->d_mappedR,             totalSz * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d->d_mappedZ,             totalSz * 3 * sizeof(float)));

        d->num_rows             = num_rows;
        d->nnz                  = nnz;
        d->numVerts             = numVerts;
        d->totalSz              = totalSz;
        d->allocated            = true;
    }

    // Store scalar parameters
    d->numLevel             = numLevel;
    d->totalNumberClusters  = totalNumberClusters;
    d->h_levelSizeY.assign(levelSizeY, levelSizeY + numLevel);

    // Upload sparse matrix (Eigen CSC = CSR for symmetric matrix)
    CUDA_CHECK(cudaMemcpy(d->d_csr_row_ptr, csc_outer_ptr,  (num_rows + 1) * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_csr_col_ind, csc_inner_idx,  nnz * sizeof(int),               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_csr_values,  csc_values,     nnz * sizeof(double),             cudaMemcpyHostToDevice));

    // Upload MAS data
    CUDA_CHECK(cudaMemcpy(d->d_invSymR,             invSymR,               invSymR_size  * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_goingNext,           goingNext,             goingNext_size * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_mapperSortedGetOrig, mapperSortedGetOriginal, numVerts * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->d_coarseTablesFlat,    coarseTablesFlat,      numVerts * 4 * sizeof(int),    cudaMemcpyHostToDevice));

    // (Re-)build cuSPARSE descriptors on every Setup call since matrix values change each frame
    if (d->mat_A)  { cusparseDestroySpMat(d->mat_A); d->mat_A = nullptr; }
    if (d->vec_p)  { cusparseDestroyDnVec(d->vec_p); d->vec_p = nullptr; }
    if (d->vec_Ap) { cusparseDestroyDnVec(d->vec_Ap); d->vec_Ap = nullptr; }
    if (d->spmv_buffer) { cudaFree(d->spmv_buffer); d->spmv_buffer = nullptr; }

    CUSPARSE_CHECK(cusparseCreateCsr(
        &d->mat_A, num_rows, num_rows, nnz,
        d->d_csr_row_ptr, d->d_csr_col_ind, d->d_csr_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnVec(&d->vec_p,  num_rows, d->d_p,  CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&d->vec_Ap, num_rows, d->d_Ap, CUDA_R_64F));

    // Pre-compute SpMV buffer size
    double alpha_dummy = 1.0, beta_dummy = 0.0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        d->cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_dummy, d->mat_A, d->vec_p, &beta_dummy, d->vec_Ap,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        &d->spmv_buffer_size));
    CUDA_CHECK(cudaMalloc(&d->spmv_buffer, d->spmv_buffer_size));
}

// ============================================================
// ApplyMasPreconditioner — orchestrates the 3 MAS kernels
// Reads d->d_r (CG residual), writes d->d_z (preconditioned residual).
// ============================================================
static void ApplyMasPreconditioner(CudaMasCgSolverImpl* d) {
    const int T = 256;

    // Zero internal MAS buffers
    CUDA_CHECK(cudaMemset(d->d_mappedR, 0, d->totalSz * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d->d_mappedZ, 0, d->totalSz * 3 * sizeof(float)));

    // Kernel 1a: level-0 vertices → d_mappedR (also atomicAdd into level-1)
    int blocks1 = (d->numVerts + T - 1) / T;
    BuildResidualHierarchyKernel<<<blocks1, T>>>(
        d->d_r, d->d_mappedR,
        d->d_mapperSortedGetOrig, d->d_goingNext,
        d->numVerts, d->numLevel);

    // Kernel 1b: level-1 → level-2 → ... (only when numLevel > 2)
    if (d->numLevel > 2 && (int)d->h_levelSizeY.size() > 2) {
        int bg = d->h_levelSizeY[1];
        int ed = d->h_levelSizeY[2];
        int count = ed - bg;
        if (count > 0) {
            int blocks1b = (count + T - 1) / T;
            PropagateResidualHierarchyKernel<<<blocks1b, T>>>(
                d->d_mappedR, d->d_goingNext,
                bg, ed, d->numLevel);
        }
    }

    // Kernel 2: per-domain local solve (independent blocks)
    int activeBlockNum = d->totalNumberClusters / 32;
    // Use 1 thread per domain block; blocks are independent
    // Batch into CUDA blocks of 128 for better SM occupancy
    int blocks2 = (activeBlockNum + 127) / 128;
    SchwarzLocalXSymKernel<<<blocks2, 128>>>(
        d->d_mappedR, d->d_mappedZ, d->d_invSymR, activeBlockNum);

    // Kernel 3: collect final z (scatter + float→double)
    int blocks3 = (d->numVerts + T - 1) / T;
    CollectFinalZKernel<<<blocks3, T>>>(
        d->d_z, d->d_mappedZ,
        d->d_mapperSortedGetOrig, d->d_coarseTablesFlat,
        d->numVerts, d->numLevel);
}

// ============================================================
// CG Solver
// ============================================================

int CudaMasCgSolver::Solve(
    const double* h_b,
    double* h_x,
    double tol,
    int max_iter,
    int check_interval)
{
    CudaMasCgSolverImpl* d = impl_;
    if (!d->allocated) {
        throw std::runtime_error("CudaMasCgSolver::Solve called before Setup()");
    }

    const int N = d->num_rows;
    const double one  =  1.0;
    const double zero =  0.0;

    // x = 0
    CUDA_CHECK(cudaMemset(d->d_x, 0, N * sizeof(double)));

    // r = b  (since x=0, initial residual = b)
    CUDA_CHECK(cudaMemcpy(d->d_r, h_b, N * sizeof(double), cudaMemcpyHostToDevice));

    // Compute ||b|| for relative convergence check
    double bnorm;
    CUBLAS_CHECK(cublasDnrm2(d->cublas_handle, N, d->d_r, 1, &bnorm));
    if (bnorm == 0.0) {
        // Trivial case: b = 0 → x = 0
        CUDA_CHECK(cudaMemset(d->d_x, 0, N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(h_x, d->d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
        return 0;
    }

    // z = M^{-1} * r
    ApplyMasPreconditioner(d);

    // p = z
    CUDA_CHECK(cudaMemcpy(d->d_p, d->d_z, N * sizeof(double), cudaMemcpyDeviceToDevice));

    // rz_old = dot(r, z)
    double rz_old;
    CUBLAS_CHECK(cublasDdot(d->cublas_handle, N, d->d_r, 1, d->d_z, 1, &rz_old));

    int final_iter = max_iter;
    for (int iter = 0; iter < max_iter; ++iter) {

        // Ap = A * p  (cuSPARSE SpMV, Ap = 1*A*p + 0*Ap)
        CUSPARSE_CHECK(cusparseSpMV(
            d->cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, d->mat_A, d->vec_p, &zero, d->vec_Ap,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d->spmv_buffer));

        // pAp = dot(p, Ap)
        double pAp;
        CUBLAS_CHECK(cublasDdot(d->cublas_handle, N, d->d_p, 1, d->d_Ap, 1, &pAp));

        if (pAp == 0.0) break;  // breakdown

        // alpha = rz_old / pAp
        double alpha = rz_old / pAp;

        // x += alpha * p
        CUBLAS_CHECK(cublasDaxpy(d->cublas_handle, N, &alpha, d->d_p, 1, d->d_x, 1));

        // r -= alpha * Ap
        double neg_alpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(d->cublas_handle, N, &neg_alpha, d->d_Ap, 1, d->d_r, 1));

        // Convergence check every check_interval iterations (GPU→CPU sync)
        if ((iter + 1) % check_interval == 0) {
            double rnorm;
            CUBLAS_CHECK(cublasDnrm2(d->cublas_handle, N, d->d_r, 1, &rnorm));
            if (rnorm <= tol * bnorm) {
                final_iter = iter + 1;
                break;
            }
        }

        // z = M^{-1} * r
        ApplyMasPreconditioner(d);

        // rz_new = dot(r, z)
        double rz_new;
        CUBLAS_CHECK(cublasDdot(d->cublas_handle, N, d->d_r, 1, d->d_z, 1, &rz_new));

        // beta = rz_new / rz_old
        double beta = rz_new / rz_old;

        // p = z + beta * p
        {
            int blocks = (N + 255) / 256;
            BetaUpdateKernel<<<blocks, 256>>>(d->d_p, d->d_z, beta, N);
        }

        // Update cuSPARSE dense vector descriptor for d_p (data pointer unchanged, just notify)
        // (Not needed since d_p device pointer doesn't change, only values change.)

        rz_old = rz_new;
    }

    // Copy solution to host
    CUDA_CHECK(cudaMemcpy(h_x, d->d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    return final_iter;
}
