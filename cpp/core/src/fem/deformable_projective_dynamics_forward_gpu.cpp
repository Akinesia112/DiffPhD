// =============================================================================
// deformable_projective_dynamics_forward_gpu.cpp
//
// Implements Algorithm 4 from:
//   "Fast But Accurate: A Real-Time Hyperelastic Simulator with
//    Robust Frictional Contact" (Zeng et al., 2025)
//
// Adds one entry point, ForwardProjectiveDynamicsAlgPhd(), dispatched from
// ForwardProjectiveDynamics() when method == "pd_eigen_alg_phd" or
// "pd_eigen_pcg_proj_gpu". Nothing already compiled in
// deformable_projective_dynamics_forward.cpp is redefined here, so the two
// translation units instantiate disjoint member functions.
//
// With CUDA_AVAILABLE the global-step SpMV and the Schur complement
// W = H S^T S H^T run on cuSPARSE/cuBLAS; the local step and CR solver stay on
// the CPU with OpenMP (hybrid design of paper §6). Without it everything falls
// back to Eigen and the method still runs as a correct CPU-only path.
// =============================================================================

#include "fem/deformable.h"
#include "friction/mesh_frictional_boundary.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <deque>
#include <ctime>

#ifdef REALSIM_SPARSE_LDLT
// RealSim SparseLDLT: truly sparse L^{-1} via METIS nested dissection
// S = sqrt(D^{-1}) * L^{-1}  →  A^{-1} = S^T * S (two SpMV)
#include "Scomponent/tools/math/SparseLDLT.h"
#include "Scomponent/tools/math/SparseMatrix.h"
#endif

// ─── Optional CUDA headers ───────────────────────────────────────────────────
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#endif
// ─────────────────────────────────────────────────────────────────────────────


// =============================================================================
// Section 4 helpers  —  Sparse inverse  S = L^{-1} I
// =============================================================================

namespace alg_phd_detail {

// =============================================================================
// GPU helpers (cuSPARSE)  —  only compiled when CUDA_AVAILABLE is defined
// =============================================================================
#ifdef CUDA_AVAILABLE

// RAII wrapper for a cuSPARSE handle.
struct CuSparseHandle {
    cusparseHandle_t h = nullptr;
    CuSparseHandle()  { cusparseCreate(&h); }
    ~CuSparseHandle() { if (h) cusparseDestroy(h); }
};

// Per-process singleton handle (thread-safe after first call).
inline cusparseHandle_t GetCuSparseHandle() {
    static CuSparseHandle inst;
    return inst.h;
}

// GPU dense vector with RAII.
struct GpuVec {
    double* d = nullptr;
    int     n = 0;
    GpuVec() = default;
    GpuVec(int n_, const double* host_data = nullptr) : n(n_) {
        cudaMalloc(&d, n * sizeof(double));
        if (host_data) cudaMemcpy(d, host_data, n*sizeof(double), cudaMemcpyHostToDevice);
    }
    ~GpuVec() { if (d) cudaFree(d); }
    void toHost(double* dst) const {
        cudaMemcpy(dst, d, n*sizeof(double), cudaMemcpyDeviceToHost);
    }
    GpuVec(const GpuVec&) = delete;
    GpuVec& operator=(const GpuVec&) = delete;
};

// GPU CSR matrix with RAII.
struct GpuCSR {
    int     rows = 0, cols = 0, nnz = 0;
    int*    d_row = nullptr;
    int*    d_col = nullptr;
    double* d_val = nullptr;
    cusparseSpMatDescr_t descr = nullptr;

    GpuCSR() = default;

    // Upload an Eigen SparseMatrix<double> (must be compressed, row-major or col-major).
    // Returns true on success, false if GPU memory allocation failed (descr stays nullptr).
    template<int Options>
    bool upload(const Eigen::SparseMatrix<double, Options>& M) {
        const_cast<Eigen::SparseMatrix<double, Options>&>(M).makeCompressed();
        rows = M.rows(); cols = M.cols(); nnz = M.nonZeros();
        if (cudaMalloc(&d_row, (rows+1)*sizeof(int))  != cudaSuccess) { d_row = nullptr; return false; }
        if (cudaMalloc(&d_col, nnz*sizeof(int))        != cudaSuccess) { d_col = nullptr; return false; }
        if (cudaMalloc(&d_val, nnz*sizeof(double))     != cudaSuccess) { d_val = nullptr; return false; }
        cudaMemcpy(d_row, M.outerIndexPtr(), (rows+1)*sizeof(int),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, M.innerIndexPtr(), nnz*sizeof(int),       cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, M.valuePtr(),      nnz*sizeof(double),    cudaMemcpyHostToDevice);
        cusparseCreateCsr(&descr, rows, cols, nnz,
                          d_row, d_col, d_val,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        return (descr != nullptr);
    }

    ~GpuCSR() {
        if (descr) cusparseDestroySpMat(descr);
        if (d_row) cudaFree(d_row);
        if (d_col) cudaFree(d_col);
        if (d_val) cudaFree(d_val);
    }
    GpuCSR(const GpuCSR&) = delete;
    GpuCSR& operator=(const GpuCSR&) = delete;
};

// y = alpha * M * x + beta * y   (cuSPARSE SpMV, CSR, double)
inline void GpuSpMV(cusparseHandle_t handle,
                    const GpuCSR& M,
                    const GpuVec& x, GpuVec& y,
                    double alpha = 1.0, double beta = 0.0) {
    cusparseDnVecDescr_t dx, dy;
    cusparseCreateDnVec(&dx, x.n, x.d, CUDA_R_64F);
    cusparseCreateDnVec(&dy, y.n, y.d, CUDA_R_64F);

    size_t buf_size = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, M.descr, dx, &beta, dy,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);
    void* buf = nullptr;
    if (buf_size > 0) cudaMalloc(&buf, buf_size);

    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, M.descr, dx, &beta, dy,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf);
    cudaDeviceSynchronize();

    if (buf) cudaFree(buf);
    cusparseDestroyDnVec(dx);
    cusparseDestroyDnVec(dy);
}

// Compute A^{-1} x = S x (single SpMV).
// ComputeSparseInverse returns S = A^{-1} directly (solver.solve(e_j)).
// S^T S would give A^{-2}, NOT A^{-1}.
inline Eigen::VectorXd AinvMultGpu(cusparseHandle_t handle,
                                    const Eigen::MatrixXd& S_dense,
                                    const Eigen::VectorXd& x_cpu) {
    int n = x_cpu.size();
    Eigen::SparseMatrix<double> S_sparse = S_dense.sparseView(1.0, 1e-14);
    S_sparse.makeCompressed();
    GpuCSR S_gpu;
    S_gpu.upload(S_sparse);
    GpuVec gx(n, x_cpu.data());
    GpuVec gy(n);
    GpuSpMV(handle, S_gpu, gx, gy);
    Eigen::VectorXd result(n);
    gy.toHost(result.data());
    return result;
}

#endif // CUDA_AVAILABLE
// =============================================================================

// Sparse inverse S = L^{-1} via column-wise sparse triangular solves; only
// columns with 2-norm above `eps` are kept. Returned dense
// (vertex_num × vertex_num per coordinate dimension).
static Eigen::MatrixXd ComputeSparseInverse(
        const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
        int n) {
    Eigen::MatrixXd S(n, n);
    Eigen::VectorXd ej(n);
    for (int j = 0; j < n; ++j) {
        ej.setZero();
        ej(j) = 1.0;
        S.col(j) = solver.solve(ej);
    }
    return S;   // S = L^{-T} (since A = L D L^T, solve gives A^{-1} e_j)
}

// Fischer-Burmeister NCP function  φ(a,b) = a + b − √(a²+b²)
// Returns phi, dPhi/da, dPhi/db.
inline void FischerBurmeister(double a, double b,
        double& phi, double& dphi_da, double& dphi_db) {
    double r    = std::sqrt(a * a + b * b);
    double eps  = 1e-14;
    if (r < eps) r = eps;
    phi     = a + b - r;
    dphi_da = 1.0 - a / r;
    dphi_db = 1.0 - b / r;
}

// Complementarity preconditioner  r_j = h² W_{jj}  (Eq. 28a of paper).
// For friction:                    r_j = h   W_{jj}  (Eq. 28b).
inline double ComputePreconditioner(double W_jj, double h, bool is_friction) {
    return is_friction ? h * W_jj : h * h * W_jj;
}

// Conjugate Residual (CR) solver for the constraint-space system
//   (Ω W Ω^T + E) Δλ = rhs_cr
// All quantities are small (num_contacts × num_contacts) so dense Eigen is fine.
static Eigen::VectorXd ConjugateResidual(
        const Eigen::MatrixXd& M,   // system matrix  (c×c)
        const Eigen::VectorXd& rhs, // right-hand side (c)
        int max_iter = 50,
        double tol   = 1e-8) {
    int c = (int)rhs.size();
    if (c == 0) return Eigen::VectorXd::Zero(0);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(c);
    Eigen::VectorXd r = rhs - M * x;
    Eigen::VectorXd p = r;
    Eigen::VectorXd Ar = M * r;
    Eigen::VectorXd Ap;

    for (int it = 0; it < max_iter; ++it) {
        Ap        = M * p;
        double rAr = r.dot(Ar);
        if (std::fabs(rAr) < 1e-30) break;
        double alpha = rAr / p.dot(Ap);
        x  += alpha * p;
        r  -= alpha * Ap;
        if (r.norm() < tol) break;
        Eigen::VectorXd Ar_new = M * r;
        double beta = Ar_new.dot(Ar_new) / rAr;
        p   = r + beta * p;
        Ar  = Ar_new;
    }
    return x;
}

// Returns pairs (i, j) of non-adjacent vertices closer than `threshold`,
// i.e. vertex i has penetrated the surface around vertex j.
static std::vector<std::pair<int,int>> DetectSelfCollisions(
    const VectorXr& q, int vertex_num, int vertex_dim,
    const std::vector<std::vector<int>>& adjacency,  // per-vertex neighbor list
    real threshold = ToReal(1e-3)) 
{
    std::vector<std::pair<int,int>> contacts;
    // Broad phase: O(n²) all-pairs (a BVH would replace this).
    for (int i = 0; i < vertex_num; ++i) {
        for (int j = i + 2; j < vertex_num; ++j) {
            // Skip adjacent vertices
            bool adjacent = false;
            for (int nb : adjacency[i]) if (nb == j) { adjacent = true; break; }
            if (adjacent) continue;

            Eigen::Matrix<real, 3, 1> pi = q.segment<3>(i * vertex_dim);
            Eigen::Matrix<real, 3, 1> pj = q.segment<3>(j * vertex_dim);
            real dist = (pi - pj).norm();
            if (dist < threshold) {
                contacts.push_back({i, j});
            }
        }
    }
    return contacts;
}

} // namespace alg_phd_detail

// =============================================================================
// Shared GPU AlgPhd A^{-1} = S^T · S cache (file scope).
//
// At file scope rather than function-local static so that
// deformable_projective_dynamics_backward_gpu.cpp can reuse the same GPU SpMV
// kernels instead of the CPU LDLT. Forward owns and populates; backward reads.
// =============================================================================
std::vector<std::vector<int>>    s_S_outer(3),  s_S_inner(3),  s_ST_outer(3), s_ST_inner(3);
std::vector<std::vector<double>> s_S_val(3),    s_ST_val(3);
std::vector<int>                 s_S_nnz(3, 0), s_ST_nnz(3, 0);
int                              s_cached_n[3] = {-1, -1, -1};
double                           s_cached_mat_key = -1.0; // fingerprint of PD energy weights
#ifdef CUDA_AVAILABLE
alg_phd_detail::GpuCSR   s_S_gpu[3];
alg_phd_detail::GpuCSR   s_ST_gpu[3];
double*               s_d_b[3]    = {nullptr, nullptr, nullptr};
double*               s_d_x[3]    = {nullptr, nullptr, nullptr};
double*               s_d_y[3]    = {nullptr, nullptr, nullptr};
cusparseDnVecDescr_t  s_desc_b[3] = {nullptr, nullptr, nullptr};
cusparseDnVecDescr_t  s_desc_y[3] = {nullptr, nullptr, nullptr};
cusparseDnVecDescr_t  s_desc_x[3] = {nullptr, nullptr, nullptr};
void*                 s_spmv1_buf[3]    = {nullptr, nullptr, nullptr};
void*                 s_spmv2_buf[3]    = {nullptr, nullptr, nullptr};
size_t                s_spmv1_buf_sz[3] = {0, 0, 0};
size_t                s_spmv2_buf_sz[3] = {0, 0, 0};
#endif

// Reports total bytes of the persistent S / S^T CSR device buffers.
// Header: cpp/core/include/fem/alg_phd_gpu_memory.h
long long AlgPhdSMemoryBytes() {
#ifdef CUDA_AVAILABLE
    long long total = 0;
    for (int d = 0; d < 3; ++d) {
        const int N = s_cached_n[d];
        if (N <= 0) continue;
        total += 4LL * (N + 1) + 4LL * s_S_nnz[d]  + 8LL * s_S_nnz[d];
        total += 4LL * (N + 1) + 4LL * s_ST_nnz[d] + 8LL * s_ST_nnz[d];
    }
    return total;
#else
    return 0;
#endif
}

// Non-zeros of one S factor (axis 0) and the cached per-axis block size N.
// Together these give the sparsity density reported in the contrast table.
long long AlgPhdSNnz()  { return static_cast<long long>(s_S_nnz[0]); }
int       AlgPhdCachedN() { return s_cached_n[0]; }

// ── Per-frame PD iteration statistics ────────────────────────────────────────
// Populated by ForwardProjectiveDynamicsAlgPhd (one entry per frame).
std::vector<int> s_frame_iters;
std::vector<int> s_frame_converged;

void AlgPhdResetFrameStats() {
    s_frame_iters.clear();
    s_frame_converged.clear();
}
std::vector<int> AlgPhdFrameIters()     { return s_frame_iters; }
std::vector<int> AlgPhdFrameConverged() { return s_frame_converged; }

#ifdef CUDA_AVAILABLE
// Shim for the backward pass (extern decl in
// deformable_projective_dynamics_backward_gpu.cpp): A^{-1}·v = S^T·(S·v) via
// two cusparseSpMV on the persistent GPU buffers. Returns false when the cache
// is not populated for this (d, vertex_num) — caller falls back to CPU LDLT.
bool AlgPhd_GpuApplyAinv(int d, int vertex_num,
                       const double* v_in, double* v_out) {
    if (d < 0 || d > 2) return false;
    if (s_cached_n[d] != vertex_num) return false;
    if (!s_S_gpu[d].descr || !s_ST_gpu[d].descr) return false;
    if (!s_d_b[d] || !s_d_y[d] || !s_d_x[d]) return false;
    cusparseHandle_t sp = alg_phd_detail::GetCuSparseHandle();
    if (!sp) return false;
    const double one = 1.0, zero = 0.0;
    if (cudaMemcpy(s_d_b[d], v_in, vertex_num * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one, s_S_gpu[d].descr, s_desc_b[d], &zero, s_desc_y[d],
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, s_spmv1_buf[d])
        != CUSPARSE_STATUS_SUCCESS) return false;
    if (cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one, s_ST_gpu[d].descr, s_desc_y[d], &zero, s_desc_x[d],
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, s_spmv2_buf[d])
        != CUSPARSE_STATUS_SUCCESS) return false;
    if (cudaMemcpy(v_out, s_d_x[d], vertex_num * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}
#endif


// =============================================================================
// Algorithm 4  —  GPU-friendly PD forward timestep
//
// Per paper §6:
//   Pre-computation (once per topology change):
//     L = Cholesky(A)
//     S = L^{-1} I                    (sparse inverse, §4.2)
//
//   Per timestep:
//     W = H S^T S H^T                 (Delassus operator,   Eq. 27)
//     q̃ = q_t + h v_t + h² M^{-1} f_ext
//     for k ∈ {0,...,n}:
//       p_i^k = project(G_i q^k)      (local step)
//       b     = M q̃ + h² Σ w_i G_i^T p_i^k
//       Evaluate Ω, J, E, g, h_vec
//       Δλ = (1/h²)(Ω W Ω^T + E)^{-1} (h_vec − J S^T S g)   (Eq. in Alg 4)
//       λ^{k+1} = λ^k + Δλ
//       q^{k+1} = S^T S (b + h² J^T λ^{k+1})
//     end
//     q_{t+h} = q^{k+1}
//     v_{t+h} = (q^{k+1} − q_t) / h
// =============================================================================
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardProjectiveDynamicsAlgPhd(
        const std::string& method,
        const VectorXr& q,
        const VectorXr& v,
        const VectorXr& a,
        const VectorXr& f_ext,
        const real dt,
        const std::map<std::string, real>& options,
        VectorXr& q_next,
        VectorXr& v_next,
        std::vector<int>& active_contact_idx) const {

    // ── Parse options ─────────────────────────────────────────────────────────
    CheckError(options.count("max_pd_iter"), "Missing option max_pd_iter.");
    CheckError(options.count("abs_tol"),     "Missing option abs_tol.");
    CheckError(options.count("rel_tol"),     "Missing option rel_tol.");
    CheckError(options.count("verbose"),     "Missing option verbose.");
    CheckError(options.count("thread_ct"),   "Missing option thread_ct.");

    const int  max_pd_iter   = static_cast<int>(options.at("max_pd_iter"));
    const int  thread_ct     = static_cast<int>(options.at("thread_ct"));
    const int  verbose_level = static_cast<int>(options.at("verbose"));
    const real abs_tol       = options.at("abs_tol");
    const real rel_tol       = options.at("rel_tol");

    // Debugging: enable detailed contact diagnostics when env var set
    bool debug_contact = false;
    const char* dbg_env = std::getenv("ALG_PHD_DEBUG_CONTACT");
    if (dbg_env && std::string(dbg_env) != "0") debug_contact = true;
    if (verbose_level >= 3) debug_contact = true;
    const int  aa_window     = options.count("aa_window")
                             ? static_cast<int>(options.at("aa_window")) : 5;

    omp_set_num_threads(thread_ct);

    // ── Pre-factorize the PD system matrix A (cached) ─────────────────────────
    // Use the standard CPU path (pd_eigen) — the Cholesky factor is already
    // managed by SetupProjectiveDynamicsSolver.
    const std::string base_method = "pd_eigen";
    SetupProjectiveDynamicsSolver(base_method, dt, options);

    const real h        = dt;
    const real mass     = element_volume_ * density_;
    const real h2m      = h * h / mass;
    const real inv_h2m  = mass / (h * h);

    // ── q̃ = q + h v + h² M^{-1} f_ext + h² M^{-1} f_state ──────────────────
    const VectorXr q_tilde = q + h * v + h2m * f_ext + h2m * ForwardStateForce(q, v);

    // New-rollout detection for the AlgPhd forward tape: within one rollout,
    // frame k+1 input q matches frame k output q_next. Otherwise start a fresh
    // tape so no stale frame cache is consumed.
    if (!alg_phd_forward_tape_.empty()) {
        const AlgPhdReplayCache& last = alg_phd_forward_tape_.back();
        bool sequential = (last.q_sol.size() == q.size());
        if (sequential) {
            const real denom = std::max<real>(
                real(1), std::max(last.q_sol.norm(), q.norm()));
            sequential = ((last.q_sol - q).norm() <= real(1e-8) * denom);
        }
        if (!sequential) alg_phd_forward_tape_.clear();
    }

    // ── Boundary conditions: structural Dirichlet only ──────────────────────────
    // Contact nodes are NOT frozen. NCP handles them via Lagrange multipliers λ.
    const std::map<int, real>& augmented_dirichlet = dirichlet_;
    const std::map<int, real>  additional_dirichlet;

    // ── Contact candidates: frictional boundary nodes near the boundary ────────
    // MeshFrictionalBoundary is pre-filtered by proximity to keep the Delassus
    // matrix small. GetDistance = surface_dist − contact_radius, so 0 means
    // "in the contact zone"; the 5×contact_radius buffer also keeps nodes that
    // will reach contact within the next few steps.
    real mesh_candidate_threshold = std::numeric_limits<real>::infinity();
    if (auto mesh_boundary = std::dynamic_pointer_cast<MeshFrictionalBoundary<vertex_dim>>(frictional_boundary_)) {
        mesh_boundary->UpdateVertices(q);
        mesh_candidate_threshold = mesh_boundary->ContactRadius() * 5.0;
    }

    std::vector<int> contact_candidates;
    for (const auto& pair : frictional_boundary_vertex_indices_) {
        const int node = pair.first;
        if (mesh_candidate_threshold < std::numeric_limits<real>::infinity()) {
            const VectorXr node_q = q.segment(node * vertex_dim, vertex_dim);
            // GetDistance = unsigned distance − contact_radius: > 0 outside the
            // contact zone, < 0 inside. Unsigned only — the face-normal sign is
            // unreliable on non-convex meshes.
            if (frictional_boundary_->GetDistance(node_q, node) > mesh_candidate_threshold)
                continue;
        }
        contact_candidates.push_back(node);
    }
    const int num_contacts = static_cast<int>(contact_candidates.size());

    // ── Contact normals + tangent basis {t1, t2} (3D) ────────────────────────
    // Normal: last column of GetLocalFrame (outward); for MeshFrictionalBoundary
    // the frame-start normal is cached as the back-face-filtering reference.
    // Tangents: first two columns of GetLocalFrame (planar boundary), or
    // Gram-Schmidt from the normal (mesh boundary). Used by NCP-integrated
    // Coulomb friction, where impulses propagate via A^{-1} J_t^T λ_t.
    std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> normals(num_contacts);
    std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> tangents1(num_contacts);
    std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> tangents2(num_contacts);
    auto mesh_boundary_ptr = std::dynamic_pointer_cast<MeshFrictionalBoundary<vertex_dim>>(frictional_boundary_);
    const bool use_mesh_anchor = static_cast<bool>(mesh_boundary_ptr);
    for (int ci = 0; ci < num_contacts; ++ci) {
        const int node = contact_candidates[ci];
        const VectorXr node_q = q.segment(node * vertex_dim, vertex_dim);
        Eigen::Matrix<double, 1, Eigen::Dynamic> n_j(1, vertex_dim);
        Eigen::Matrix<double, 1, Eigen::Dynamic> t1_j = Eigen::Matrix<double, 1, Eigen::Dynamic>::Zero(1, vertex_dim);
        Eigen::Matrix<double, 1, Eigen::Dynamic> t2_j = Eigen::Matrix<double, 1, Eigen::Dynamic>::Zero(1, vertex_dim);
        if (use_mesh_anchor) {
            Eigen::Matrix<real, vertex_dim, 1> closest, n_vec;
            mesh_boundary_ptr->GetClosestPointAndNormal(node_q, node, closest, n_vec);
            for (int d = 0; d < vertex_dim; ++d) n_j(0, d) = static_cast<double>(n_vec(d));
            if constexpr (vertex_dim == 3) {
                // Gram-Schmidt: pick axis least parallel to n, then cross product
                Eigen::Matrix<double, 3, 1> n(n_j(0,0), n_j(0,1), n_j(0,2));
                int axis = 0;
                for (int d = 1; d < 3; ++d)
                    if (std::abs(n(d)) < std::abs(n(axis))) axis = d;
                Eigen::Matrix<double, 3, 1> e = Eigen::Matrix<double, 3, 1>::Zero(); e(axis) = 1.0;
                const Eigen::Matrix<double, 3, 1> t1 = (e - n.dot(e) * n).normalized();
                const Eigen::Matrix<double, 3, 1> t2 = n.cross(t1).normalized();
                for (int d = 0; d < 3; ++d) { t1_j(0,d) = t1(d); t2_j(0,d) = t2(d); }
            }
        } else {
            const auto frame = frictional_boundary_->GetLocalFrame(node_q, node);
            for (int d = 0; d < vertex_dim; ++d)
                n_j(0, d) = static_cast<double>(frame(d, vertex_dim - 1));
            if constexpr (vertex_dim == 3) {
                for (int d = 0; d < vertex_dim; ++d) {
                    t1_j(0, d) = static_cast<double>(frame(d, 0));
                    t2_j(0, d) = static_cast<double>(frame(d, 1));
                }
            }
        }
        normals[ci] = n_j;
        tangents1[ci] = t1_j;
        tangents2[ci] = t2_j;
    }

    if (debug_contact) {
        // Frame-start one-line summary: which nodes are candidates and how
        // close they start. (Per-node detail only when verbose is set.)
        const char* dbg_v = std::getenv("ALG_PHD_DEBUG_CONTACT_VERBOSE");
        const bool verbose_init = dbg_v && std::string(dbg_v) != "0";
        double init_min = std::numeric_limits<double>::infinity();
        double init_min_neg = 0.0;  // most negative (deepest penetration)
        for (int ci = 0; ci < num_contacts; ++ci) {
            const int node = contact_candidates[ci];
            const VectorXr node_q0 = q.segment(node * vertex_dim, vertex_dim);
            const double d0 = static_cast<double>(frictional_boundary_->GetDistance(node_q0, node));
            init_min = std::min(init_min, d0);
            init_min_neg = std::min(init_min_neg, d0);
            if (verbose_init) {
                std::cerr << "[ALG_PHD_DEBUG] init_distance[node="<< node << "]=" << d0 << std::endl;
                std::cerr << "[ALG_PHD_DEBUG] normal[node="<< node << "]=";
                for (int d = 0; d < vertex_dim; ++d) std::cerr << normals[ci](0,d) << (d+1<vertex_dim?",":"\n");
            }
        }
        std::cerr << "[ALG_PHD_DEBUG] frame_start: num_contacts=" << num_contacts
                  << " min_signed_dist=" << (num_contacts>0?init_min:0.0)
                  << " deepest_penetration=" << init_min_neg << std::endl;
    }

    // ── Sparse inverse S = L^{-1}  (per coordinate dimension) ────────────────
    // We compute S for the per-dimension system (vertex_num × vertex_num)
    // and store the STS product (= A^{-1} per dimension).
    const int vertex_num = mesh_.NumOfVertices();

    // ── Timing helpers ────────────────────────────────────────────────────────
    auto wall_ms = []() -> double {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
    };
    static double t_build_S = 0, t_delassus = 0, t_ainv_b = 0, t_q_update = 0;
    static int    n_frames = 0, n_iters = 0;
    int frame_iters = 0;

    // ── Sparse S = A^{-1} (cached, RealSim §4.2) ─────────────────────────────
    // S is built column-by-column via LDLT solve and stored as raw CSR (RealSim
    // format, avoiding Eigen CSC/CSR confusion on the GPU); on GPU it is
    // uploaded as GpuCSR and the solve is S*b, one cusparseSpMV per dim.
    // Keyed by vertex_num, so it is rebuilt only on topology change.
    // State lives at file scope — see the top of this TU.
#ifdef CUDA_AVAILABLE
    cusparseHandle_t sp_handle = alg_phd_detail::GetCuSparseHandle();
    const bool use_gpu = (sp_handle != nullptr);
    // CUDA_VISIBLE_DEVICES remaps physical GPU to logical 0; always use device 0.
    if (use_gpu) cudaSetDevice(0);
#else
    const bool use_gpu = false;
#endif

    // Fingerprint A = inv_h2m*I + K_pd to detect material changes across
    // different deformable instances (same topology, different E/nu).
    double mat_key = element_volume_ * density_ / (dt * dt); // inv_h2m
    for (const auto& pair : pd_element_energies_) mat_key += pair.first->stiffness();
    if (mat_key != s_cached_mat_key) {
        for (int d = 0; d < vertex_dim; ++d) s_cached_n[d] = -1;
        s_cached_mat_key = mat_key;
    }

    for (int d = 0; d < vertex_dim; ++d) {
        if (s_cached_n[d] != vertex_num) {
            double _t0_build = wall_ms();
            if (verbose_level > 0)
                std::cout << "[AlgPhd] Building sparse S dim=" << d
                          << " N=" << vertex_num << std::endl;

#ifdef REALSIM_SPARSE_LDLT
            // ── RealSim path: S = sqrt(D^{-1}) * L^{-1} (truly sparse) ──────
            // Uses METIS nested dissection → L^{-1} sparsity from etree
            // Mirrors SparseInverseSolver::computeLowerInverse_fast()
            {
                // A is taken straight from pd_lhs_[d] (the matrix handed to
                // SetupProjectiveDynamicsSolver). pd_lhs_ is Eigen column-major
                // (CSC) and RealSim needs row-major (CSR); A is symmetric, so
                // CSR(A) = CSC(Aᵀ) = CSC(A).transpose() in Eigen.
                Eigen::SparseMatrix<double> A_csc = pd_lhs_[d].template cast<double>();
                A_csc.makeCompressed();
                Eigen::SparseMatrix<double, Eigen::RowMajor> A(A_csc);
                A.makeCompressed();

                RealSim::tools::linearalgebra::CSMatrix csrA(A.rows(), A.cols(), A.nonZeros(),
                    A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr());

                RealSim::tools::linearalgebra::SparseLDLTData ldlt;
                ldlt.factorize(csrA);

                // sqrt(D^{-1}) scaling
                Eigen::VectorXd sqrtDinv(vertex_num);
                for (int i = 0; i < vertex_num; ++i)
                    sqrtDinv(i) = std::sqrt(ldlt.invD[i]);

                // Compute L^{-1} (sparse, etree structure)
                ldlt.computeLowerInverse();

                // getS_CSFormat() returns L^{-1} in CSC → reinterpreted as CSR of
                // L^{-T}. S^T = L^{-T} · Diag(sqrtDinv), so element
                // (row i, col j=innerInd[p]) scales by sqrtDinv[j]; scaling by
                // sqrtDinv[i] would give Diag(sqrtDinv)·L^{-T} ≠ S^T.
                auto csrST = ldlt.getS_CSFormat();
                for (int i = 0; i < vertex_num; ++i)
                    for (int p = csrST._outerPtr[i]; p < csrST._outerPtr[i+1]; ++p)
                        csrST._values[p] *= sqrtDinv(csrST._innerInd[p]);  // col j, not row i

                // S = (S^T)^T (column-major)
                auto csrS = csrST.switchOrder();

                // Store raw CSR data for direct GPU upload
                // Only store S^T in CSR (csrST). Use TRANSPOSE op for S*v.
                s_ST_outer[d].assign(csrST._outerPtr.begin(), csrST._outerPtr.end());
                s_ST_inner[d].assign(csrST._innerInd.begin(), csrST._innerInd.end());
                s_ST_val[d].assign(csrST._values.begin(), csrST._values.end());
                s_ST_nnz[d] = csrST._nnz;
                s_S_nnz[d] = csrST._nnz;  // same matrix, different op
            }
#else
            s_S_nnz[d] = 0; s_ST_nnz[d] = 0;
#endif
            s_cached_n[d] = vertex_num;

            // ── Verify: S^T * S * v == A^{-1} * v (first build, any N) ──────
            if (true) {
                Eigen::VectorXd v_test = Eigen::VectorXd::Random(vertex_num);
                Eigen::VectorXd ainv_ref = pd_eigen_solver_[d].solve(v_test);

                // CPU: S^T * (S * v_test) using raw CSR
                Eigen::Map<const Eigen::SparseMatrix<double,Eigen::RowMajor>> ST_map(
                    vertex_num, vertex_num, s_ST_nnz[d],
                    s_ST_outer[d].data(), s_ST_inner[d].data(), s_ST_val[d].data());
                // S * v = ST^T * v = ST_map.transpose() * v
                Eigen::VectorXd Sv  = ST_map.transpose() * v_test;
                Eigen::VectorXd STSv = ST_map * Sv;

                double err = (STSv - ainv_ref).norm() / ainv_ref.norm();
                // std::cout << "[AlgPhd-VERIFY] dim=" << d
                //           << " |S^T*S*v - A^{-1}*v| / |A^{-1}*v| = " << err
                //           << (err < 1e-6 ? "  OK" : "  WRONG!") << std::endl;
            }
            // ────────────────────────────────────────────────────────────────

            t_build_S += wall_ms() - _t0_build;
            double sparsity = s_S_nnz[d] > 0 ? 100.0 * s_S_nnz[d] / ((double)vertex_num * vertex_num) : 0.0;
            // std::cout << "[AlgPhd-PERF] S dim=" << d
            //           << " N=" << vertex_num
            //           << " nnz=" << s_S_nnz[d]
            //           << " density=" << sparsity << "%"
            //           << " build=" << t_build_S << "ms" << std::endl;
#ifdef CUDA_AVAILABLE
            if (use_gpu && s_S_nnz[d] > 0) {
                // cuSPARSE NON_TRANSPOSE is wrong on this driver version while
                // TRANSPOSE works, so both S and S^T are uploaded and each step
                // uses TRANSPOSE:  y = S*v → TRANSPOSE(S^T),  x = S^T*y → TRANSPOSE(S).
                // setFromTriplets is needed to get sorted RowMajor CSR: the LDLT
                // etree innerInd order is not guaranteed sorted.

                // --- Upload S^T in RowMajor CSR (sorted) ---
                Eigen::SparseMatrix<double,Eigen::RowMajor> ST_rm(vertex_num, vertex_num);
                {
                    std::vector<Eigen::Triplet<double>> trips;
                    trips.reserve(s_ST_nnz[d]);
                    for (int i = 0; i < vertex_num; ++i)
                        for (int p = s_ST_outer[d][i]; p < s_ST_outer[d][i+1]; ++p)
                            trips.emplace_back(i, s_ST_inner[d][p], s_ST_val[d][p]);
                    ST_rm.setFromTriplets(trips.begin(), trips.end());
                }
                bool st_ok = s_ST_gpu[d].upload(ST_rm);
                if (!st_ok) {
                    std::cout << "[AlgPhd] dim=" << d << " GPU S^T upload OOM → CPU fallback" << std::endl;
                    continue;  // skip GPU setup for this dim
                }

                // --- Upload S = (S^T)^T in RowMajor CSR (sorted) ---
                Eigen::SparseMatrix<double,Eigen::RowMajor> S_rm(vertex_num, vertex_num);
                {
                    std::vector<Eigen::Triplet<double>> trips;
                    trips.reserve(s_ST_nnz[d]);
                    // S[j,i] = S^T[i,j]  (transpose of S^T data)
                    for (int i = 0; i < vertex_num; ++i)
                        for (int p = s_ST_outer[d][i]; p < s_ST_outer[d][i+1]; ++p)
                            trips.emplace_back(s_ST_inner[d][p], i, s_ST_val[d][p]);
                    S_rm.setFromTriplets(trips.begin(), trips.end());
                }
                bool s_ok = s_S_gpu[d].upload(S_rm);
                if (!s_ok) {
                    std::cout << "[AlgPhd] dim=" << d << " GPU S upload OOM → CPU fallback" << std::endl;
                    continue;
                }

                // ── Pre-allocate persistent GPU buffers & SpMV workspaces ──────────────
                // Mirrors RealSim CUDASparseInverseSolver::send_to_GPU(): cuda_b,
                // cuda_y, cuda_x allocated once, workspace buffers pre-computed.
                if (s_d_b[d]) { cudaFree(s_d_b[d]); s_d_b[d] = nullptr; }
                if (s_d_y[d]) { cudaFree(s_d_y[d]); s_d_y[d] = nullptr; }
                if (s_d_x[d]) { cudaFree(s_d_x[d]); s_d_x[d] = nullptr; }
                if (cudaMalloc(&s_d_b[d], vertex_num*sizeof(double)) != cudaSuccess ||
                    cudaMalloc(&s_d_y[d], vertex_num*sizeof(double)) != cudaSuccess ||
                    cudaMalloc(&s_d_x[d], vertex_num*sizeof(double)) != cudaSuccess) {
                    std::cout << "[AlgPhd] dim=" << d << " GPU vector bufs OOM → CPU fallback" << std::endl;
                    s_d_b[d] = s_d_y[d] = s_d_x[d] = nullptr; continue;
                }
                if (s_desc_b[d]) { cusparseDestroyDnVec(s_desc_b[d]); s_desc_b[d] = nullptr; }
                if (s_desc_y[d]) { cusparseDestroyDnVec(s_desc_y[d]); s_desc_y[d] = nullptr; }
                if (s_desc_x[d]) { cusparseDestroyDnVec(s_desc_x[d]); s_desc_x[d] = nullptr; }
                cusparseCreateDnVec(&s_desc_b[d], vertex_num, s_d_b[d], CUDA_R_64F);
                cusparseCreateDnVec(&s_desc_y[d], vertex_num, s_d_y[d], CUDA_R_64F);
                cusparseCreateDnVec(&s_desc_x[d], vertex_num, s_d_x[d], CUDA_R_64F);
                {
                    double one_pre=1.0, zero_pre=0.0;
                    size_t sz1=0, sz2=0;
                    cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one_pre, s_S_gpu[d].descr, s_desc_b[d], &zero_pre, s_desc_y[d],
                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &sz1);
                    cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one_pre, s_ST_gpu[d].descr, s_desc_y[d], &zero_pre, s_desc_x[d],
                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &sz2);
                    if (s_spmv1_buf[d]) { cudaFree(s_spmv1_buf[d]); s_spmv1_buf[d] = nullptr; }
                    if (s_spmv2_buf[d]) { cudaFree(s_spmv2_buf[d]); s_spmv2_buf[d] = nullptr; }
                    s_spmv1_buf_sz[d] = sz1;
                    s_spmv2_buf_sz[d] = sz2;
                    // Always alloc ≥1 byte so cuSPARSE never receives a NULL workspace
                    if (cudaMalloc(&s_spmv1_buf[d], std::max(sz1, (size_t)1)) != cudaSuccess ||
                        cudaMalloc(&s_spmv2_buf[d], std::max(sz2, (size_t)1)) != cudaSuccess) {
                        std::cout << "[AlgPhd] dim=" << d << " GPU workspace OOM → CPU fallback" << std::endl;
                        s_spmv1_buf[d] = s_spmv2_buf[d] = nullptr; continue;
                    }
                }

                // GPU VERIFY deferred to after all dims are built (see post-loop check below)
            }
#endif
        }
    }

    // ── GPU VERIFY (post-loop, first build only) ──────────────────────────────
#ifdef CUDA_AVAILABLE
    {
        static bool s_verified = false;
        if (use_gpu && !s_verified) {
            s_verified = true;
            // Clear sticky CUDA errors from S upload / workspace alloc phase.
            cudaGetLastError(); cudaDeviceSynchronize(); cudaGetLastError();
            for (int d = 0; d < vertex_dim; ++d) {
                if (s_S_nnz[d] == 0 || s_S_gpu[d].descr == nullptr ||
                    s_d_b[d] == nullptr || s_spmv1_buf[d] == nullptr) {
                    std::cout << "[AlgPhd-GPU-VERIFY] dim=" << d << "  CPU (upload OOM)" << std::endl;
                    continue;
                }
                Eigen::VectorXd v_test = Eigen::VectorXd::Random(vertex_num);
                Eigen::VectorXd ainv_ref = pd_eigen_solver_[d].solve(v_test);
                double one2=1.0, zero2=0.0;
                cudaMemcpy(s_d_b[d], v_test.data(), vertex_num*sizeof(double), cudaMemcpyHostToDevice);
                cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one2, s_S_gpu[d].descr, s_desc_b[d], &zero2, s_desc_y[d],
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, s_spmv1_buf[d]);
                cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one2, s_ST_gpu[d].descr, s_desc_y[d], &zero2, s_desc_x[d],
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, s_spmv2_buf[d]);
                cudaError_t sync_err = cudaDeviceSynchronize();
                if (sync_err != cudaSuccess) {
                    // Kernel fault (e.g., invalid d_val due to partial OOM) → fall back to CPU
                    std::cout << "[AlgPhd-GPU-VERIFY] dim=" << d
                              << " kernel error " << sync_err << " → CPU fallback" << std::endl;
                    if (s_S_gpu[d].descr)  { cusparseDestroySpMat(s_S_gpu[d].descr);  s_S_gpu[d].descr  = nullptr; }
                    if (s_ST_gpu[d].descr) { cusparseDestroySpMat(s_ST_gpu[d].descr); s_ST_gpu[d].descr = nullptr; }
                    cudaGetLastError();
                    continue;
                }
                Eigen::VectorXd gpu_res(vertex_num);
                cudaMemcpy(gpu_res.data(), s_d_x[d], vertex_num*sizeof(double), cudaMemcpyDeviceToHost);
                double gpu_err = (gpu_res - ainv_ref).norm() / ainv_ref.norm();
                std::cout << "[AlgPhd-GPU-VERIFY] dim=" << d
                          << " GPU S^T*S*v err=" << gpu_err
                          << (gpu_err < 1e-6 ? "  OK" : "  WRONG!") << std::endl;
            }
        }
    }
#endif

    // A^{-1} v = S^T*(S*v) via two GPU NON_TRANSPOSE SpMV (aligned with RealSim solve_vec_gpu).
    // Pre-allocated persistent buffers — no malloc/free per call.
    //   Step 1: y = NON_TRANSPOSE(s_S_gpu)  * v = S   * v
    //   Step 2: x = NON_TRANSPOSE(s_ST_gpu) * y = S^T * y
    auto Ainv_vec = [&](int d, const Eigen::VectorXd& v) -> Eigen::VectorXd {
#ifdef CUDA_AVAILABLE
        if (use_gpu && s_S_nnz[d] > 0 && s_cached_n[d] == vertex_num
            && s_S_gpu[d].descr != nullptr && s_ST_gpu[d].descr != nullptr
            && s_d_b[d] != nullptr && s_d_y[d] != nullptr && s_d_x[d] != nullptr) {
            double one=1.0, zero=0.0;
            // Upload input vector to persistent GPU buffer
            cudaMemcpy(s_d_b[d], v.data(), vertex_num*sizeof(double), cudaMemcpyHostToDevice);
            // Step 1: y = S * v  (NON_TRANSPOSE on s_S_gpu)
            cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, s_S_gpu[d].descr, s_desc_b[d], &zero, s_desc_y[d], CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, s_spmv1_buf[d]);
            // Step 2: x = S^T * y  (NON_TRANSPOSE on s_ST_gpu)
            cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, s_ST_gpu[d].descr, s_desc_y[d], &zero, s_desc_x[d], CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, s_spmv2_buf[d]);
            cudaDeviceSynchronize();
            Eigen::VectorXd x(vertex_num);
            cudaMemcpy(x.data(), s_d_x[d], vertex_num*sizeof(double), cudaMemcpyDeviceToHost);
            return x;
        }
#endif
        return pd_eigen_solver_[d].solve(v);
    };

    // ── Delassus W + pre-compute Ainv_cols ───────────────────────────────────
    // Ainv_cols[d][ci] = A_d^{-1} * (n_ci[d] * e_{node_ci}), computed once per
    // timestep (for W) and reused in every PD iter → 1 fewer solve/iter (−50%).
    //
    // GPU batch path (contacts > 0): RHS_batch is [vertex_num × num_contacts]
    // per dim with column ci = normals[ci](0,d) * e_{node_ci}; two cuSPARSE SpMM
    // calls replace num_contacts SpMVs, and Ainv_cols[d][ci] = X.col(ci).
    //     Y = S   * RHS_batch   (NON_TRANSPOSE on s_S_gpu)
    //     X = S^T * Y           (NON_TRANSPOSE on s_ST_gpu)
    double _t0_del = wall_ms();
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(num_contacts, num_contacts);
    std::vector<std::vector<Eigen::VectorXd>> Ainv_cols(vertex_dim,
        std::vector<Eigen::VectorXd>(num_contacts));

    bool batch_ok = false;
#ifdef CUDA_AVAILABLE
    // Process-wide kill switch. Some cuSPARSE 12 builds report SUCCESS for the
    // batched SpMM but leave the output unwritten → W = 0 → degenerate contact
    // system → λ saturates at its clamp → the solid explodes on first contact.
    // The first use is checked against the per-column A^{-1} path; on failure
    // the batched path is never used again.
    static bool s_batch_spmm_broken = false;
    const bool can_batch = use_gpu && num_contacts > 0
                        && !s_batch_spmm_broken
                        && s_S_gpu[0].descr != nullptr && s_ST_gpu[0].descr != nullptr
                        && s_cached_n[0] == vertex_num;
    if (can_batch) {
        batch_ok = true;
        for (int d = 0; d < vertex_dim && batch_ok; ++d) {
            const int K = num_contacts;
            const int N = vertex_num;

            // Build RHS_batch on CPU (column-major, N × K)
            std::vector<double> rhs_cpu(N * K, 0.0);
            for (int ci = 0; ci < K; ++ci) {
                const int ni = contact_candidates[ci];
                rhs_cpu[ci * N + ni] = static_cast<double>(normals[ci](0, d));
            }

            // GPU buffers: RHS, Y (= S*RHS), X (= S^T*Y)
            double *d_RHS = nullptr, *d_Y = nullptr, *d_X = nullptr;
            void *buf1 = nullptr, *buf2 = nullptr;
            cusparseDnMatDescr_t desc_RHS = nullptr, desc_Y = nullptr, desc_X = nullptr;
            std::vector<double> x_cpu(N * K);
            bool ok = true;

            const double one = 1.0, zero = 0.0;
            size_t sz1 = 0, sz2 = 0;

            ok = ok && cudaMalloc(&d_RHS, N * K * sizeof(double)) == cudaSuccess;
            ok = ok && cudaMalloc(&d_Y,   N * K * sizeof(double)) == cudaSuccess;
            ok = ok && cudaMalloc(&d_X,   N * K * sizeof(double)) == cudaSuccess;
            ok = ok && cudaMemcpy(d_RHS, rhs_cpu.data(), N * K * sizeof(double),
                                  cudaMemcpyHostToDevice) == cudaSuccess;

            // Dense-matrix descriptors (column-major, leading dim = N)
            ok = ok && cusparseCreateDnMat(&desc_RHS, N, K, N, d_RHS, CUDA_R_64F,
                                           CUSPARSE_ORDER_COL) == CUSPARSE_STATUS_SUCCESS;
            ok = ok && cusparseCreateDnMat(&desc_Y,   N, K, N, d_Y,   CUDA_R_64F,
                                           CUSPARSE_ORDER_COL) == CUSPARSE_STATUS_SUCCESS;
            ok = ok && cusparseCreateDnMat(&desc_X,   N, K, N, d_X,   CUDA_R_64F,
                                           CUSPARSE_ORDER_COL) == CUSPARSE_STATUS_SUCCESS;

            // NOTE: the algorithm MUST be CUSPARSE_SPMM_CSR_ALG2 (deterministic
            // CSR path). With CUSPARSE_SPMM_ALG_DEFAULT, cuSPARSE 12.3 returns
            // CUSPARSE_STATUS_SUCCESS for this CSR × column-major-dense product
            // yet leaves the output essentially unwritten: W ≡ 0, r_prec at its
            // 1e-10 floor, M_sys singular, Δλ = M_sys^{-1}·rhs saturating the
            // 1e8 λ clamp on first contact. The check below guards regressions.
            //
            // SpMM #1: Y = S * RHS  (NON_TRANSPOSE on s_S_gpu[d])
            ok = ok && cusparseSpMM_bufferSize(sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, s_S_gpu[d].descr, desc_RHS, &zero, desc_Y,
                CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2, &sz1) == CUSPARSE_STATUS_SUCCESS;
            // CUSPARSE 12 rejects a null workspace even when bufferSize == 0
            // (the SpMV path above hits the same trap) → always allocate ≥ 1 byte.
            ok = ok && cudaMalloc(&buf1, sz1 > 0 ? sz1 : 1) == cudaSuccess;
            ok = ok && cusparseSpMM(sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, s_S_gpu[d].descr, desc_RHS, &zero, desc_Y,
                CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2, buf1) == CUSPARSE_STATUS_SUCCESS;

            // SpMM #2: X = S^T * Y  (NON_TRANSPOSE on s_ST_gpu[d])
            ok = ok && cusparseSpMM_bufferSize(sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, s_ST_gpu[d].descr, desc_Y, &zero, desc_X,
                CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2, &sz2) == CUSPARSE_STATUS_SUCCESS;
            ok = ok && cudaMalloc(&buf2, sz2 > 0 ? sz2 : 1) == cudaSuccess;
            ok = ok && cusparseSpMM(sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, s_ST_gpu[d].descr, desc_Y, &zero, desc_X,
                CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2, buf2) == CUSPARSE_STATUS_SUCCESS;

            ok = ok && cudaDeviceSynchronize() == cudaSuccess;

            // Download result and extract columns into Ainv_cols[d][ci]
            ok = ok && cudaMemcpy(x_cpu.data(), d_X, N * K * sizeof(double),
                                  cudaMemcpyDeviceToHost) == cudaSuccess;
            if (ok) {
                for (int ci = 0; ci < K; ++ci) {
                    Ainv_cols[d][ci] = Eigen::Map<const Eigen::VectorXd>(
                        x_cpu.data() + ci * N, N);
                }
            }

            // Cleanup
            if (desc_RHS) cusparseDestroyDnMat(desc_RHS);
            if (desc_Y)   cusparseDestroyDnMat(desc_Y);
            if (desc_X)   cusparseDestroyDnMat(desc_X);
            if (d_RHS) cudaFree(d_RHS);
            if (d_Y)   cudaFree(d_Y);
            if (d_X)   cudaFree(d_X);
            if (buf1)  cudaFree(buf1);
            if (buf2)  cudaFree(buf2);

            if (!ok) {
                // Clear the sticky error so later CUDA calls are not poisoned.
                cudaGetLastError();
                batch_ok = false;
                s_batch_spmm_broken = true;
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    PrintWarning("AlgPhd: batched SpMM for the Delassus operator failed"
                                 " → falling back to per-column A^{-1}.");
                }
            }
        }

        // One-time correctness check of the batched result against the trusted
        // per-column path. Cheap: a single extra A^{-1} solve, once per process.
        if (batch_ok) {
            static bool verified = false;
            if (!verified) {
                verified = true;
                const int d_chk = vertex_dim - 1;
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(vertex_num);
                rhs(contact_candidates[0]) =
                    static_cast<double>(normals[0](0, d_chk));
                const Eigen::VectorXd ref = Ainv_vec(d_chk, rhs);
                const double ref_norm = ref.norm();
                const double err = (Ainv_cols[d_chk][0] - ref).norm()
                                 / std::max(ref_norm, 1e-300);
                if (!(err < 1e-8)) {
                    s_batch_spmm_broken = true;
                    batch_ok = false;
                    std::cerr << "[AlgPhd] batched SpMM Delassus verification FAILED"
                              << " (rel_err=" << err << ")"
                              << " → using per-column A^{-1} for all later frames."
                              << std::endl;
                }
            }
        }
    }
#endif
    if (!batch_ok) {
        for (int d = 0; d < vertex_dim; ++d) {
            for (int ci = 0; ci < num_contacts; ++ci) {
                const int ni = contact_candidates[ci];
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(vertex_num);
                rhs(ni) = static_cast<double>(normals[ci](0, d));
                Ainv_cols[d][ci] = Ainv_vec(d, rhs);
            }
        }
    }
    for (int ci = 0; ci < num_contacts; ++ci)
        for (int cj = ci; cj < num_contacts; ++cj) {
            const int nj = contact_candidates[cj];
            double w = 0.0;
            for (int d = 0; d < vertex_dim; ++d)
                w += normals[cj](0,d) * Ainv_cols[d][ci](nj);
            W(ci, cj) = w; W(cj, ci) = w;
        }
    t_delassus += wall_ms() - _t0_del;
    ++n_frames;
    // if (n_frames == 1 || n_frames % 100 == 0) {
    //     std::cout << "[AlgPhd-PERF] frame=" << n_frames
    //               << " num_contacts=" << num_contacts
    //               << " delassus=" << (t_delassus/n_frames) << "ms/frame" << std::endl;
    // }

    // ── Friction options (parsed once, used in NCP loop and post-loop) ────────
    const real friction_mu_val = options.count("friction_mu") ? options.at("friction_mu") : real(0);
    const real friction_kf_val = options.count("friction_kf") ? options.at("friction_kf") : real(0);
    const real restitution_val = options.count("restitution")  ? options.at("restitution")  : real(0);
    // NCP-integrated Coulomb friction: enabled in 3D when friction_mu > 0
    const bool do_friction_ncp = (vertex_dim == 3) && (num_contacts > 0) && (friction_mu_val > 0);

    // ── Tangential Ainv_cols + W_tt diagonal blocks (CPU, 3D Coulomb friction) ─
    // Ainv_t{1,2}_cols[d][ci] = A_d^{-1} · (t{1,2}_ci[d] * e_node_ci): the elastic
    // response of every node to a unit tangential force at contact node ci, i.e.
    // friction propagates through the whole body.
    // W_tt_blocks[ci] = 2×2 diagonal block of the tangential Delassus operator.
    std::vector<std::vector<Eigen::VectorXd>> Ainv_t1_cols(vertex_dim, std::vector<Eigen::VectorXd>(num_contacts));
    std::vector<std::vector<Eigen::VectorXd>> Ainv_t2_cols(vertex_dim, std::vector<Eigen::VectorXd>(num_contacts));
    std::vector<Eigen::Matrix2d> W_tt_blocks(num_contacts, Eigen::Matrix2d::Zero());
    if constexpr (vertex_dim == 3) {
        if (do_friction_ncp) {
            for (int d = 0; d < vertex_dim; ++d) {
                for (int ci = 0; ci < num_contacts; ++ci) {
                    const int ni = contact_candidates[ci];
                    Eigen::VectorXd rhs1 = Eigen::VectorXd::Zero(vertex_num);
                    Eigen::VectorXd rhs2 = Eigen::VectorXd::Zero(vertex_num);
                    rhs1(ni) = tangents1[ci](0, d);
                    rhs2(ni) = tangents2[ci](0, d);
                    Ainv_t1_cols[d][ci] = Ainv_vec(d, rhs1);
                    Ainv_t2_cols[d][ci] = Ainv_vec(d, rhs2);
                }
            }
            for (int ci = 0; ci < num_contacts; ++ci) {
                const int ni = contact_candidates[ci];
                double w11 = 0.0, w12 = 0.0, w22 = 0.0;
                for (int d = 0; d < vertex_dim; ++d) {
                    w11 += tangents1[ci](0,d) * Ainv_t1_cols[d][ci](ni);
                    w12 += tangents1[ci](0,d) * Ainv_t2_cols[d][ci](ni);
                    w22 += tangents2[ci](0,d) * Ainv_t2_cols[d][ci](ni);
                }
                W_tt_blocks[ci] << w11, w12, w12, w22;
            }
        }
    }

    // ── Complementarity preconditioner r_j = h^2 * W_{jj}  (Eq. 28a) ───────────
    Eigen::VectorXd r_prec(num_contacts);
    for (int ci = 0; ci < num_contacts; ++ci)
        r_prec(ci) = std::max(static_cast<double>(h * h) * W(ci, ci), 1e-10);

    // ── PD iteration ─────────────────────────────────────────────────────────
    VectorXr q_sol = q;
    VectorXr q_prev = q;
    // Lambda warm-start: reuse previous frame's multipliers for nodes that
    // remain candidates, cutting NCP iterations from ~1000 to ~250 on
    // sustained contact. New candidates start at 0.
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(num_contacts);
    if (!alg_phd_prev_candidates_.empty() && alg_phd_prev_lambda_.size() == static_cast<int>(alg_phd_prev_candidates_.size())) {
        std::unordered_map<int,int> prev_idx;
        for (int i = 0; i < static_cast<int>(alg_phd_prev_candidates_.size()); ++i)
            prev_idx[alg_phd_prev_candidates_[i]] = i;
        for (int ci = 0; ci < num_contacts; ++ci) {
            auto it = prev_idx.find(contact_candidates[ci]);
            if (it != prev_idx.end())
                lambda(ci) = alg_phd_prev_lambda_(it->second);
        }
    }
    // Tangential multipliers for NCP-integrated Coulomb friction (no warm-start)
    Eigen::VectorXd lambda_t1 = Eigen::VectorXd::Zero(num_contacts);
    Eigen::VectorXd lambda_t2 = Eigen::VectorXd::Zero(num_contacts);
    // Last-iteration values cached for backward tape.
    Eigen::VectorXd omega_last = Eigen::VectorXd::Zero(num_contacts);

    // Anderson Acceleration history (Type-II Walker & Ni 2011)
    std::deque<VectorXr> aa_q_hist, aa_g_hist;
    VectorXr prev_pd_rhs = VectorXr::Zero(dofs_);

    bool converged = false;
    for (int iter = 0; iter < max_pd_iter; ++iter) {

        // ── Local step: p_i^k = project(G_i q^k) ─────────────────────────────
        const VectorXr pd_rhs_vec = ProjectiveDynamicsLocalStep(q_sol, a, augmented_dirichlet);
        // Standard PD RHS: projective local step only, no ElasticForce. Adding
        // ElasticForce(q_sol) brings extra stiffness K_elastic whose spectral
        // radius can exceed λ_min(A), so the fixed point diverges for stiff
        // Neohookean materials. CPU pd_eigen escapes this via BFGS line search;
        // this fixed point needs a contractive map, so ElasticForce is omitted.
        const VectorXr b_full = inv_h2m * q_tilde + pd_rhs_vec;

        VectorXr b_bc = b_full;
        for (const auto& pair : augmented_dirichlet) b_bc(pair.first) = 0;

        q_prev = q_sol;

        if (auto mesh_boundary = std::dynamic_pointer_cast<MeshFrictionalBoundary<vertex_dim>>(frictional_boundary_)) {
            mesh_boundary->UpdateVertices(q_sol);
        }

        if (num_contacts > 0) {
            Eigen::VectorXd delta_n(num_contacts);
            for (int ci = 0; ci < num_contacts; ++ci) {
                const int node = contact_candidates[ci];
                if (use_mesh_anchor) {
                    // The frame-start normal is the reference that filters out
                    // back-side faces, which would flip the contact direction on
                    // deep penetration. UpdateVertices still refreshes the
                    // opposing body each iter, so the moving surface is tracked.
                    Eigen::Matrix<real, vertex_dim, 1> ref_n;
                    for (int d = 0; d < vertex_dim; ++d) ref_n(d) = static_cast<real>(normals[ci](0, d));
                    delta_n(ci) = static_cast<double>(
                        mesh_boundary_ptr->GetSignedDistanceWithReferenceNormal(
                            q_sol.segment(node * vertex_dim, vertex_dim), node, ref_n));
                } else {
                    delta_n(ci) = static_cast<double>(
                        frictional_boundary_->GetDistance(
                            q_sol.segment(node * vertex_dim, vertex_dim), node));
                }
            }

            // Verbose per-iter dump only when ALG_PHD_DEBUG_CONTACT_VERBOSE=1.
            const char* dbg_verbose_env = std::getenv("ALG_PHD_DEBUG_CONTACT_VERBOSE");
            const bool debug_contact_verbose = dbg_verbose_env && std::string(dbg_verbose_env) != "0";
            if (debug_contact && debug_contact_verbose) {
                std::cerr << "[ALG_PHD_DEBUG] iter=" << iter << " delta_n: ";
                for (int ci = 0; ci < num_contacts; ++ci) std::cerr << delta_n(ci) << (ci+1<num_contacts?",":"\n");
                std::cerr << "[ALG_PHD_DEBUG] W_diag: ";
                for (int ci = 0; ci < num_contacts; ++ci) std::cerr << W(ci,ci) << (ci+1<num_contacts?",":"\n");
                std::cerr << "[ALG_PHD_DEBUG] lambda_before: ";
                for (int ci = 0; ci < num_contacts; ++ci) std::cerr << lambda(ci) << (ci+1<num_contacts?",":"\n");
            }

            Eigen::VectorXd omega(num_contacts), E_diag(num_contacts);
            for (int ci = 0; ci < num_contacts; ++ci) {
                double dj = delta_n(ci), lj = lambda(ci), rj = r_prec(ci);
                double denom = std::sqrt(dj*dj + rj*rj*lj*lj);
                if (denom < 1e-14) denom = 1e-14;
                omega(ci)  = 1.0 - dj / denom;
                E_diag(ci) = (1.0 - rj*lj / denom) * rj;
            }
            omega_last = omega;

            // M_sys = Ω W Ω^T + E 
            Eigen::MatrixXd M_sys(num_contacts, num_contacts);
            for (int ci = 0; ci < num_contacts; ++ci)
                for (int cj = 0; cj < num_contacts; ++cj)
                    M_sys(ci, cj) = omega(ci) * W(ci, cj) * omega(cj);
            for (int ci = 0; ci < num_contacts; ++ci)
                M_sys(ci, ci) += E_diag(ci);

            // g = b_full + J^T λ^k  (normal + tangential contributions)
            const Eigen::VectorXd omega_lambda = omega.array() * lambda.array();
            VectorXr JT_lambda = VectorXr::Zero(dofs_);
            for (int ci = 0; ci < num_contacts; ++ci) {
                const int node = contact_candidates[ci];
                for (int d = 0; d < vertex_dim; ++d) {
                    JT_lambda(node * vertex_dim + d) +=
                        static_cast<real>(normals[ci](0,d) * omega_lambda(ci));
                    if constexpr (vertex_dim == 3) {
                        JT_lambda(node * vertex_dim + d) +=
                            static_cast<real>(tangents1[ci](0,d) * lambda_t1(ci)
                                            + tangents2[ci](0,d) * lambda_t2(ci));
                    }
                }
            }
            const VectorXr g_vec = b_full + JT_lambda;

            // h_vec_j = -φ_j + J_n q^k 
            Eigen::VectorXd h_vec(num_contacts);
            for (int ci = 0; ci < num_contacts; ++ci) {
                double dj=delta_n(ci), lj=lambda(ci), rj=r_prec(ci);
                double denom = std::sqrt(dj*dj + rj*rj*lj*lj);
                if (denom < 1e-14) denom = 1e-14;
                double phi_j = dj + rj*lj - denom;

                const int node = contact_candidates[ci];
                double H_q = 0.0;
                for (int d = 0; d < vertex_dim; ++d)
                    H_q += normals[ci](0,d) * static_cast<double>(q_sol(node * vertex_dim + d));
                h_vec(ci) = -phi_j + omega(ci) * H_q; 
            }

            // ── Reuse strategy: ONE solve per dim (Ainv_b), reuse Ainv_cols ────
            // Ainv_g = A^{-1}b + Σ_ci ω_ci λ_ci     * Ainv_cols[d][ci]
            // q^{k+1}= A^{-1}b + Σ_ci ω_ci λ_new_ci * Ainv_cols[d][ci]
            // → no second solve after the λ update
            double _t0_ainvb = wall_ms();
            std::vector<Eigen::VectorXd> Ainv_b(vertex_dim);
            {
                const auto b_reshape = Eigen::Map<const Eigen::Matrix<real,vertex_dim,-1>>(
                    b_full.data(), vertex_dim, vertex_num);
                for (int d = 0; d < vertex_dim; ++d)
                    Ainv_b[d] = Ainv_vec(d, b_reshape.row(d).template cast<double>());
            }
            t_ainv_b += wall_ms() - _t0_ainvb;

            VectorXr Ainv_g = VectorXr::Zero(dofs_);
            for (int d = 0; d < vertex_dim; ++d) {
                Eigen::VectorXd Ainv_gd = Ainv_b[d];
                for (int ci = 0; ci < num_contacts; ++ci) {
                    Ainv_gd += omega(ci) * lambda(ci) * Ainv_cols[d][ci];
                    if constexpr (vertex_dim == 3) {
                        if (do_friction_ncp) {
                            Ainv_gd += lambda_t1(ci) * Ainv_t1_cols[d][ci];
                            Ainv_gd += lambda_t2(ci) * Ainv_t2_cols[d][ci];
                        }
                    }
                }
                for (int ni = 0; ni < vertex_num; ++ni)
                    Ainv_g(ni * vertex_dim + d) = static_cast<real>(Ainv_gd(ni));
            }

            Eigen::VectorXd J_Ainv_g(num_contacts);
            for (int ci = 0; ci < num_contacts; ++ci) {
                const int node = contact_candidates[ci];
                double val = 0.0;
                for (int d = 0; d < vertex_dim; ++d)
                    val += normals[ci](0,d) * static_cast<double>(Ainv_g(node * vertex_dim + d));
                J_Ainv_g(ci) = omega(ci) * val;
            }

            const Eigen::VectorXd rhs_cr = h_vec - J_Ainv_g;
            lambda += M_sys.ldlt().solve(rhs_cr);
            lambda = lambda.cwiseMax(0.0).cwiseMin(1e8);

            if (debug_contact && debug_contact_verbose) {
                std::cerr << "[ALG_PHD_DEBUG] lambda_after: ";
                for (int ci = 0; ci < num_contacts; ++ci) std::cerr << lambda(ci) << (ci+1<num_contacts?",":"\n");
                int n_active = 0; for (int ci=0; ci<num_contacts; ++ci) if (lambda(ci) > 0) ++n_active;
                std::cerr << "[ALG_PHD_DEBUG] active_count=" << n_active << std::endl;
            }

            // Precompute omega_lambda_new (old omega × new lambda) for q update and tangential solve
            const Eigen::VectorXd omega_lambda_new = omega.array() * lambda.array();

            // ── NCP-integrated Coulomb friction: tangential cone projection (3D) ─
            // Per contact node: normal-only q_sol → tangential displacement →
            // 2×2 sticking solve → projection onto the Coulomb cone.
            if constexpr (vertex_dim == 3) {
                if (do_friction_ncp) {
                    for (int ci = 0; ci < num_contacts; ++ci) {
                        if (lambda(ci) <= 0.0) { lambda_t1(ci) = lambda_t2(ci) = 0.0; continue; }
                        const int node = contact_candidates[ci];
                        // Normal-only q_sol at contact node (Ainv_b + Σ_cj ω_cj λ_new_cj Ainv_n_cj)
                        Eigen::Matrix<double, 3, 1> q_n_node = Eigen::Matrix<double, 3, 1>::Zero();
                        for (int d = 0; d < 3; ++d) {
                            q_n_node(d) = Ainv_b[d](node);
                            for (int cj = 0; cj < num_contacts; ++cj)
                                q_n_node(d) += omega_lambda_new(cj) * Ainv_cols[d][cj](node);
                        }
                        // Step displacement (from q at start of timestep)
                        Eigen::Matrix<double, 3, 1> step;
                        for (int d = 0; d < 3; ++d)
                            step(d) = q_n_node(d) - static_cast<double>(q(node * 3 + d));
                        // Project to tangential subspace
                        const Eigen::Matrix<double, 3, 1> n_hat(normals[ci](0,0), normals[ci](0,1), normals[ci](0,2));
                        const Eigen::Matrix<double, 3, 1> u_t_vec = step - step.dot(n_hat) * n_hat;
                        const Eigen::Matrix<double, 3, 1> t1_hat(tangents1[ci](0,0), tangents1[ci](0,1), tangents1[ci](0,2));
                        const Eigen::Matrix<double, 3, 1> t2_hat(tangents2[ci](0,0), tangents2[ci](0,1), tangents2[ci](0,2));
                        const double u_t1 = u_t_vec.dot(t1_hat);
                        const double u_t2 = u_t_vec.dot(t2_hat);
                        // Sticking solve: W_tt * λ_t = -[u_t1; u_t2]  (2×2)
                        const Eigen::Matrix2d& Wtt = W_tt_blocks[ci];
                        const double det = Wtt(0,0)*Wtt(1,1) - Wtt(0,1)*Wtt(1,0);
                        Eigen::Vector2d lam_t_stick;
                        if (std::abs(det) > 1e-20)
                            lam_t_stick = Wtt.inverse() * Eigen::Vector2d(-u_t1, -u_t2);
                        else
                            lam_t_stick.setZero();
                        // Coulomb cone projection: |λ_t| ≤ μ λ_n
                        const double cone_r = static_cast<double>(friction_mu_val) * lambda(ci);
                        if (lam_t_stick.norm() <= cone_r + 1e-14) {
                            lambda_t1(ci) = lam_t_stick(0);
                            lambda_t2(ci) = lam_t_stick(1);
                        } else {
                            const double ut_mag = std::sqrt(u_t1*u_t1 + u_t2*u_t2);
                            if (ut_mag > 1e-14) {
                                lambda_t1(ci) = -cone_r * u_t1 / ut_mag;
                                lambda_t2(ci) = -cone_r * u_t2 / ut_mag;
                            } else {
                                lambda_t1(ci) = lambda_t2(ci) = 0.0;
                            }
                        }
                    }
                }
            }

            // q^{k+1} = Ainv_b + Σ_ci [ω_ci λ_n_ci Ainv_n_ci + λ_t1_ci Ainv_t1_ci + λ_t2_ci Ainv_t2_ci]
            double _t0_qu = wall_ms();
            for (int d = 0; d < vertex_dim; ++d) {
                Eigen::VectorXd q_d = Ainv_b[d];
                for (int ci = 0; ci < num_contacts; ++ci) {
                    q_d += omega_lambda_new(ci) * Ainv_cols[d][ci];
                    if constexpr (vertex_dim == 3) {
                        if (do_friction_ncp) {
                            q_d += lambda_t1(ci) * Ainv_t1_cols[d][ci];
                            q_d += lambda_t2(ci) * Ainv_t2_cols[d][ci];
                        }
                    }
                }
                for (int ni = 0; ni < vertex_num; ++ni)
                    q_sol(ni * vertex_dim + d) = static_cast<real>(q_d(ni));
            }
            t_q_update += wall_ms() - _t0_qu;
            ++n_iters; ++frame_iters;
        } else {
            // No contacts: q = A^{-1} b = S^T*(S*b) (one Ainv_vec per dim)
            double _t0_ainvb_nc = wall_ms();
            const auto b_reshape = Eigen::Map<const Eigen::Matrix<real,vertex_dim,-1>>(
                b_bc.data(), vertex_dim, vertex_num);
            for (int d = 0; d < vertex_dim; ++d) {
                const Eigen::VectorXd q_d = Ainv_vec(d, b_reshape.row(d).template cast<double>());
                for (int ni = 0; ni < vertex_num; ++ni)
                    q_sol(ni * vertex_dim + d) = static_cast<real>(q_d(ni));
            }
            t_ainv_b += wall_ms() - _t0_ainvb_nc;
            ++n_iters; ++frame_iters;
        }

        // Enforce structural Dirichlet BCs (raw PD result)
        for (const auto& pair : augmented_dirichlet)
            q_sol(pair.first) = pair.second;

        // ── Anderson Acceleration (Type-II, Walker & Ni 2011) ────────────────
        // Mixes recent iterates to accelerate the PD fixed point.
        // Reg = 1e-6 * ||dG||_F² / m (Walker & Ni scale; 4 orders above the old
        // 1e-10, which let γ blow up when residuals became collinear).
        // If ||γ|| > 10 the extrapolation is unreliable → clear history and keep
        // the raw fixed-point step for this iteration.
        if (aa_window > 0) {
            const VectorXr g_k = q_sol - q_prev;
            aa_q_hist.push_back(q_prev);
            aa_g_hist.push_back(g_k);
            const int m = static_cast<int>(aa_g_hist.size()) - 1;
            if (m >= 1) {
                Eigen::MatrixXd dG(dofs_, m), dQ(dofs_, m);
                for (int i = 0; i < m; ++i) {
                    dG.col(i) = (aa_g_hist[i + 1] - aa_g_hist[i]).template cast<double>();
                    dQ.col(i) = (aa_q_hist[i + 1] - aa_q_hist[i]).template cast<double>();
                }
                Eigen::MatrixXd GtG = dG.transpose() * dG;
                const double reg = std::max(1e-6 * dG.squaredNorm() / std::max(m, 1), 1e-12);
                GtG.diagonal().array() += reg;
                Eigen::VectorXd gamma = GtG.ldlt().solve(
                    dG.transpose() * g_k.template cast<double>());
                Eigen::VectorXd q_aa_candidate =
                    (q_prev + g_k).template cast<double>() - (dQ + dG) * gamma;

                if (gamma.norm() >= 10.0) {
                    // AA extrapolation unreliable — restart history, keep raw PD step
                    aa_q_hist.clear();
                    aa_g_hist.clear();
                } else {
                    q_sol = q_aa_candidate.template cast<real>();
                    for (const auto& pair : augmented_dirichlet)
                        q_sol(pair.first) = pair.second;
                }
            }
            while (static_cast<int>(aa_q_hist.size()) > aa_window) {
                aa_q_hist.pop_front();
                aa_g_hist.pop_front();
            }
        }

        // ── Convergence check: ||Δq|| AND ||Δ pd_rhs|| ──────────────────────
        const real abs_err       = (q_sol - q_prev).norm();
        const real rhs_norm      = q_prev.norm();
        const real pd_rhs_change = (pd_rhs_vec - prev_pd_rhs).norm();
        const real pd_rhs_ref    = prev_pd_rhs.norm();
        prev_pd_rhs = pd_rhs_vec;
        if (verbose_level > 0)
            std::cout << "alg_phd PD iter " << iter
                      << "  Δq=" << abs_err
                      << "  Δpd_rhs=" << pd_rhs_change
                      << std::endl;
        const bool q_step_small = (abs_err       <= rel_tol * rhs_norm   + abs_tol);
        const bool pd_rhs_small = (pd_rhs_change <= rel_tol * pd_rhs_ref + abs_tol);
        if (iter > 0 && q_step_small && pd_rhs_small) {
            converged = true;
            break;
        }
    }

    // Record per-frame solver statistics (read from Python via
    // AlgPhdFrameIters/AlgPhdFrameConverged for the contrast-sweep table).
    s_frame_iters.push_back(frame_iters);
    s_frame_converged.push_back(converged ? 1 : 0);

    if (!converged) {
        std::cout << "frame: " << n_frames << " ";
        PrintWarning("alg_phd: PD loop did not converge.");
    }

    // Per-frame contact diagnostic (one line): whether the NCP solver produced
    // meaningful contact forces and whether penetration was resolved by frame
    // end. ALG_PHD_DEBUG_CONTACT_VERBOSE=1 adds per-iter detail.
    if (debug_contact && num_contacts > 0) {
        double lam_max = 0.0, lam_sum = 0.0;
        int n_active = 0;
        for (int ci = 0; ci < num_contacts; ++ci) {
            const double lj = lambda(ci);
            if (lj > 1e-12) ++n_active;
            lam_sum += lj;
            if (lj > lam_max) lam_max = lj;
        }
        double dn_min = std::numeric_limits<double>::infinity();
        double dn_max = -std::numeric_limits<double>::infinity();
        for (int ci = 0; ci < num_contacts; ++ci) {
            const int node = contact_candidates[ci];
            double dn;
            if (use_mesh_anchor) {
                Eigen::Matrix<real, vertex_dim, 1> ref_n;
                for (int d = 0; d < vertex_dim; ++d) ref_n(d) = static_cast<real>(normals[ci](0, d));
                dn = static_cast<double>(mesh_boundary_ptr->GetSignedDistanceWithReferenceNormal(
                    q_sol.segment(node * vertex_dim, vertex_dim), node, ref_n));
            } else {
                dn = static_cast<double>(frictional_boundary_->GetDistance(
                    q_sol.segment(node * vertex_dim, vertex_dim), node));
            }
            if (dn < dn_min) dn_min = dn;
            if (dn > dn_max) dn_max = dn;
        }
        // Inactive candidates with negative delta_n (should be active but aren't)
        int n_inactive_negative = 0;
        double inactive_dn_min = 0.0;
        for (int ci = 0; ci < num_contacts; ++ci) {
            if (lambda(ci) > 1e-12) continue;
            const int node = contact_candidates[ci];
            double dn;
            if (use_mesh_anchor) {
                Eigen::Matrix<real, vertex_dim, 1> ref_n;
                for (int d = 0; d < vertex_dim; ++d) ref_n(d) = static_cast<real>(normals[ci](0, d));
                dn = static_cast<double>(mesh_boundary_ptr->GetSignedDistanceWithReferenceNormal(
                    q_sol.segment(node * vertex_dim, vertex_dim), node, ref_n));
            } else {
                dn = static_cast<double>(frictional_boundary_->GetDistance(
                    q_sol.segment(node * vertex_dim, vertex_dim), node));
            }
            if (dn < -1e-12) { ++n_inactive_negative; inactive_dn_min = std::min(inactive_dn_min, dn); }
        }
        std::cerr << "[ALG_PHD_DEBUG] frame_end: n_contacts=" << num_contacts
                  << " active=" << n_active
                  << " inactive_but_penetrating=" << n_inactive_negative
                  << " inactive_dn_min=" << inactive_dn_min
                  << " lam_max=" << lam_max
                  << " lam_sum=" << lam_sum
                  << " end_min_signed_dist=" << dn_min
                  << " end_max_signed_dist=" << dn_max
                  << " iters=" << (converged ? "converged" : "max")
                  << std::endl;
    }

    // Per-frame timing summary
    // if (n_frames > 0 && (n_frames == 1 || n_frames % 100 == 0)) {
    //     std::cout << "[AlgPhd-PERF] frame=" << n_frames
    //               << " pd_iters=" << frame_iters
    //               << " ainv_b=" << (t_ainv_b/std::max(1,n_iters)) << "ms/iter"
    //               << " q_update=" << (t_q_update/std::max(1,n_iters)) << "ms/iter"
    //               << " delassus=" << (t_delassus/n_frames) << "ms/frame"
    //               << " GPU=" << (use_gpu ? "ON" : "OFF")
    //               << std::endl;
    // }

    // ── Restitution (post-NCP velocity correction) ───────────────────────────
    // Coulomb friction is handled inside the NCP loop (NCP-integrated elastic
    // impulses A^{-1} J_t^T λ_t), so only restitution is left post-loop.
    {
        const bool do_restitution = restitution_val > 0;
        if (do_restitution) {
            const real default_contact_slop = use_mesh_anchor
                ? static_cast<real>(0.1) * static_cast<real>(mesh_boundary_ptr->ContactRadius())
                : real(1e-8);
            const real friction_contact_slop = options.count("friction_contact_slop")
                                   ? options.at("friction_contact_slop") : default_contact_slop;

            for (int ci = 0; ci < num_contacts; ++ci) {
                if (lambda(ci) <= 0) continue;
                const int node = contact_candidates[ci];
                const Eigen::Matrix<real, vertex_dim, 1> x_node = q_sol.segment(node * vertex_dim, vertex_dim);
                Eigen::Matrix<real, vertex_dim, 1> n_hat;
                real signed_dist;
                if (use_mesh_anchor) {
                    Eigen::Matrix<real, vertex_dim, 1> ref_n;
                    for (int d = 0; d < vertex_dim; ++d) ref_n(d) = static_cast<real>(normals[ci](0, d));
                    signed_dist = static_cast<real>(
                        mesh_boundary_ptr->GetSignedDistanceWithReferenceNormal(x_node, node, ref_n));
                    Eigen::Matrix<real, vertex_dim, 1> closest, n_cur;
                    mesh_boundary_ptr->GetClosestPointAndNormal(x_node, node, closest, n_cur);
                    if (n_cur.dot(ref_n) < 0) n_cur = -n_cur;
                    n_hat = n_cur;
                } else {
                    signed_dist = static_cast<real>(frictional_boundary_->GetDistance(x_node, node));
                    n_hat = frictional_boundary_->GetNormal(x_node, node);
                }
                const real n_norm = n_hat.norm();
                if (n_norm <= real(1e-12)) continue;
                n_hat /= n_norm;
                if (signed_dist > friction_contact_slop) continue;

                const real v_in_n = v.segment(node * vertex_dim, vertex_dim).dot(n_hat);
                if (v_in_n < 0) {
                    Eigen::Matrix<real, vertex_dim, 1> v_post;
                    for (int d = 0; d < vertex_dim; ++d)
                        v_post(d) = (q_sol(node * vertex_dim + d) - q(node * vertex_dim + d)) / h;
                    const real v_out_n  = v_post.dot(n_hat);
                    const real v_target = -restitution_val * v_in_n;
                    const real delta_v  = v_target - v_out_n;
                    if (delta_v > 0) {
                        for (int d = 0; d < vertex_dim; ++d)
                            q_sol(node * vertex_dim + d) += h * delta_v * n_hat(d);
                    }
                }
            }
        }
    }

    q_next = q_sol;
    v_next = (q_next - q) / h;

    // Update active_contact_idx: nodes with positive contact force (lambda > 0)
    active_contact_idx.clear();
    for (int ci = 0; ci < num_contacts; ++ci) {
        if (lambda(ci) > 0)
            active_contact_idx.push_back(contact_candidates[ci]);
    }

    // Save lambda and candidates for warm-starting the next frame.
    alg_phd_prev_candidates_ = contact_candidates;
    alg_phd_prev_lambda_ = lambda;

    // Lightweight per-frame AlgPhd replay cache: lets the backward fast path
    // skip replaying the fixed active-set contact solve.
    AlgPhdReplayCache frame_cache;
    frame_cache.contact_candidates = active_contact_idx;
    const int num_active = static_cast<int>(active_contact_idx.size());
    frame_cache.q_tilde = q_tilde;
    frame_cache.q_sol = q_next;
    frame_cache.b_full = VectorXr::Zero(dofs_);
    frame_cache.g_vec = VectorXr::Zero(dofs_);
    frame_cache.Ainv_g = VectorXr::Zero(dofs_);
    frame_cache.lambda = Eigen::VectorXd::Zero(num_active);
    frame_cache.omega = Eigen::VectorXd::Zero(num_active);
    frame_cache.E_diag = Eigen::VectorXd::Zero(num_active);
    frame_cache.delta_n = Eigen::VectorXd::Zero(num_active);
    frame_cache.W = Eigen::MatrixXd::Zero(num_active, num_active);
    frame_cache.M_sys = Eigen::MatrixXd::Zero(num_active, num_active);
    frame_cache.normals.resize(num_active);
    frame_cache.STS.resize(vertex_dim);

    if (num_contacts > 0 && num_active > 0) {
        std::unordered_map<int, int> candidate_idx;
        candidate_idx.reserve(contact_candidates.size());
        for (int ci = 0; ci < static_cast<int>(contact_candidates.size()); ++ci)
            candidate_idx[contact_candidates[ci]] = ci;

        for (int ai = 0; ai < num_active; ++ai) {
            const int node = active_contact_idx[ai];
            const auto it = candidate_idx.find(node);
            if (it == candidate_idx.end()) {
                frame_cache.normals[ai] =
                    Eigen::Matrix<real, vertex_dim, 1>::Zero();
                continue;
            }
            const int ci = it->second;
            Eigen::Matrix<real, vertex_dim, 1> ncol;
            for (int d = 0; d < vertex_dim; ++d)
                ncol(d) = static_cast<real>(normals[ci](0, d));
            frame_cache.normals[ai] = ncol;
            if (ci < omega_last.size()) frame_cache.omega(ai) = omega_last(ci);
            if (ci < lambda.size()) frame_cache.lambda(ai) = lambda(ci);
        }
    }
    alg_phd_forward_tape_.push_back(std::move(frame_cache));
    constexpr size_t kMaxAlgPhdTapeFrames = 8192;
    if (alg_phd_forward_tape_.size() > kMaxAlgPhdTapeFrames)
        alg_phd_forward_tape_.erase(alg_phd_forward_tape_.begin(),
                                 alg_phd_forward_tape_.begin() + (alg_phd_forward_tape_.size() - kMaxAlgPhdTapeFrames));
}

// =============================================================================
// Explicit template instantiations  (match those in the original forward file)
// =============================================================================
template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
