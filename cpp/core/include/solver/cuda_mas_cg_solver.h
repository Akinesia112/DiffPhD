#pragma once
// cuda_mas_cg_solver.h — Host-side API for the GPU CG solver with MAS preconditioner.
//
// This header is intentionally free of CUDA types, SSE/AVX intrinsics, and Eigen types
// so that it can be included by both the host C++ compiler (GCC) and nvcc without
// conflicts. All CUDA-specific types are hidden behind the PIMPL pattern in the .cu file.

#include <vector>

// Forward declaration of implementation struct (full definition in .cu file).
struct CudaMasCgSolverImpl;

class CudaMasCgSolver {
public:
    CudaMasCgSolver();
    ~CudaMasCgSolver();

    // Upload the sparse matrix (Newton Hessian, symmetric — Eigen CSC reinterpreted as CSR)
    // and all MAS preconditioner data to GPU. Call once per backward-pass frame.
    //
    // Sparse matrix parameters (Eigen SparseMatrix<double>, column-major / CSC):
    //   num_rows        — matrix dimension (= 3*numVerts)
    //   nnz             — number of nonzeros
    //   csc_outer_ptr   — outerIndexPtr()  [num_rows+1]  (reinterpreted as CSR row_ptr for symmetric mat)
    //   csc_inner_idx   — innerIndexPtr()  [nnz]
    //   csc_values      — valuePtr()       [nnz]
    //
    // MAS preconditioner parameters (from SeSchwarzPreconditioner after PreparePreconditioner):
    //   invSymR                  — compact L^{-T}D^{-1}L^{-1} per block [invSymR_size floats]
    //   goingNext                — absolute index of parent at next coarser level [goingNext_size ints]
    //   mapperSortedGetOriginal  — Morton-sorted index -> original vertex index [numVerts ints]
    //   coarseTablesFlat         — flattened Int4 coarseTables [numVerts*4 ints]
    //   levelSizeX               — m_levelSize[l].x (count at level l) [numLevel ints]
    //   levelSizeY               — m_levelSize[l].y (prefix begin at level l) [numLevel ints]
    //   numLevel                 — number of hierarchy levels
    //   totalSz                  — total nodes across all levels (m_totalSz)
    //   totalNumberClusters      — total domain blocks across all levels
    void Setup(
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
        int totalNumberClusters
    );

    // Solve op * x = b using PCG with MAS preconditioner entirely on GPU.
    // h_b and h_x are host (CPU) pointers of length num_rows.
    // Returns the number of CG iterations performed.
    // check_interval: check convergence every N iterations (avoids per-iteration GPU->CPU sync).
    int Solve(
        const double* h_b,
        double* h_x,
        double tol,
        int max_iter,
        int check_interval = 50
    );

private:
    // PIMPL: all CUDA handles, device pointers, and descriptors live in CudaMasCgSolverImpl.
    CudaMasCgSolverImpl* impl_;

    // Non-copyable.
    CudaMasCgSolver(const CudaMasCgSolver&) = delete;
    CudaMasCgSolver& operator=(const CudaMasCgSolver&) = delete;
};
