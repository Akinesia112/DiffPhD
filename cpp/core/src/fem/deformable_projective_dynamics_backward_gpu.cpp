// =============================================================================
// deformable_projective_dynamics_backward_gpu.cpp
//
// Backward pass (adjoint / reverse-mode AD) for Algorithm 4.
//
// Keeps the PD backbone adjoint and adds a fixed-active-set contact adjoint
// correction consistent with the forward AlgPhd linearization:
//   q_{k+1} = A^{-1}( b + J^T (Omega lambda) )
//   lambda  updated by M_sys = Omega W Omega^T + E
//
// Scope: the active set is fixed (given by active_contact_idx) and the nonsmooth
// derivative of active-set switching is NOT modeled, but the multiplier feedback
// through the active-contact block is.
// =============================================================================

#include "fem/deformable.h"
#include "friction/mesh_frictional_boundary.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"
#include <cmath>
#include <vector>
#include <algorithm>
#include <deque>

// ─── Forward-declared shared AlgPhd GPU A^{-1} cache ────────────────────────────
// Populated by ForwardProjectiveDynamicsAlgPhd() at file scope in
// deformable_projective_dynamics_forward_gpu.cpp; reused here for ~1 ms GPU
// SpMV instead of ~10 ms CPU LDLT per solve. Reached through the shims below,
// which fall back to pd_eigen_solver_[d].solve() when the cache is not ready.
extern int s_cached_n[3];
#ifdef CUDA_AVAILABLE
// Shim: returns true iff forward has uploaded S, S^T for this (d, n) and
// writes A^{-1}·v_in into v_out via two SpMV. False ⇒ caller must CPU-solve.
bool AlgPhd_GpuApplyAinv(int d, int vertex_num,
                       const double* v_in, double* v_out);
#endif

// ─── Optional CUDA headers ───────────────────────────────────────────────────
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#endif
// ─────────────────────────────────────────────────────────────────────────────

namespace alg_phd_bwd_detail {

// Same dense sparse-inverse helper as forward AlgPhd: dense A^{-1} per
// coordinate block.
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
    return S;   // A^{-1}
}

} // namespace alg_phd_bwd_detail

namespace alg_phd_exact_detail {

inline void ComputeOmegaAndE(
        double delta, double lambda, double r,
        double& omega, double& E,
        double& domega_ddelta, double& domega_dlambda,
        double& dE_ddelta, double& dE_dlambda) {
    double denom = std::sqrt(delta * delta + r * r * lambda * lambda);
    if (denom < 1e-14) denom = 1e-14;

    omega = 1.0 - delta / denom;
    E     = (1.0 - r * lambda / denom) * r;

    const double denom3 = denom * denom * denom;
    domega_ddelta  = -(r * r * lambda * lambda) / denom3;
    domega_dlambda =  (delta * r * r * lambda) / denom3;
    dE_ddelta      =  (r * r * lambda * delta) / denom3;
    dE_dlambda     = -r * (1.0 / denom - (r * r * lambda * lambda) / denom3);
}

inline void ComputePhiAndDerivative(
        double delta, double lambda, double r,
        double& phi, double& dphi_ddelta, double& dphi_dlambda) {
    double denom = std::sqrt(delta * delta + r * r * lambda * lambda);
    if (denom < 1e-14) denom = 1e-14;

    phi          = delta + r * lambda - denom;
    dphi_ddelta  = 1.0 - delta / denom;
    dphi_dlambda = r - (r * r * lambda) / denom;
}

template<int vertex_dim>
Eigen::Matrix<real, vertex_dim, 1> ApplyDnDx(
        const std::vector<Eigen::Matrix<real, vertex_dim, 1>>& dn_dx,
        const Eigen::Matrix<real, vertex_dim, 1>& dx) {
    Eigen::Matrix<real, vertex_dim, 1> out =
        Eigen::Matrix<real, vertex_dim, 1>::Zero();
    for (int k = 0; k < vertex_dim; ++k) {
        out += dn_dx[k] * dx(k);
    }
    return out;
}

} // namespace alg_phd_exact_detail



template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ReplayAlgPhdFixedActiveSet(
        const VectorXr& q,
        const VectorXr& v,
        const VectorXr& a,
        const VectorXr& f_ext,
        const real dt,
        const std::vector<int>& active_contact_idx,
        const std::map<std::string, real>& options,
        AlgPhdReplayCache& cache) const {

    CheckError(options.count("max_pd_iter"), "Missing option max_pd_iter.");
    CheckError(options.count("abs_tol"),     "Missing option abs_tol.");
    CheckError(options.count("rel_tol"),     "Missing option rel_tol.");

    const int  max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol     = options.at("abs_tol");
    const real rel_tol     = options.at("rel_tol");

    const real h       = dt;
    const real mass    = element_volume_ * density_;
    const real h2m     = h * h / mass;
    const real inv_h2m = mass / (h * h);

    const int vertex_num = mesh_.NumOfVertices();

    // GPU A^{-1}·v via the S, S^T cached by forward (forward_gpu.cpp file-scope
    // state); falls back to CPU LDLT when that cache is not ready — first call
    // without a prior forward, or CUDA disabled.
    auto apply_ainv = [&](int d, const Eigen::VectorXd& v) -> Eigen::VectorXd {
        Eigen::VectorXd out(vertex_num);
#ifdef CUDA_AVAILABLE
        if (AlgPhd_GpuApplyAinv(d, vertex_num, v.data(), out.data())) return out;
#endif
        return pd_eigen_solver_[d].solve(v);
    };

    cache.contact_candidates = active_contact_idx;
    const int num_active = static_cast<int>(cache.contact_candidates.size());

    // Fast path: no contacts → replay is unused by backbone adjoint, skip the PD loop.
    if (num_active == 0) {
        cache.q_sol  = q;
        cache.lambda = Eigen::VectorXd::Zero(0);
        cache.omega  = Eigen::VectorXd::Zero(0);
        cache.E_diag = Eigen::VectorXd::Zero(0);
        cache.delta_n= Eigen::VectorXd::Zero(0);
        cache.M_sys  = Eigen::MatrixXd::Zero(0, 0);
        cache.W      = Eigen::MatrixXd::Zero(0, 0);
        cache.Ainv_g = VectorXr::Zero(dofs_);
        cache.normals.clear();
        cache.STS.resize(vertex_dim);
        cache.q_tilde = q + h * v + h2m * f_ext + h2m * ForwardStateForce(q, v);
        cache.b_full  = VectorXr::Zero(dofs_);
        cache.g_vec   = VectorXr::Zero(dofs_);
        return;
    }

    cache.q_sol     = q;
    cache.lambda    = Eigen::VectorXd::Zero(num_active);
    cache.omega     = Eigen::VectorXd::Zero(num_active);
    cache.E_diag    = Eigen::VectorXd::Zero(num_active);
    cache.delta_n   = Eigen::VectorXd::Zero(num_active);
    cache.M_sys     = Eigen::MatrixXd::Zero(num_active, num_active);
    cache.W         = Eigen::MatrixXd::Zero(num_active, num_active);
    cache.Ainv_g    = VectorXr::Zero(dofs_);
    cache.normals.resize(num_active);
    cache.STS.resize(vertex_dim);

    cache.q_tilde = q + h * v + h2m * f_ext + h2m * ForwardStateForce(q, v);

    // ── Contact normals + Delassus W ──────────────────────────────────────────
    // W[ci][cj] = Σ_d n_ci[d] * (A_d^{-1})[node_ci, node_cj] * n_cj[d]
    //           = Σ_d n_ci[d] * ainv_col_ci_d(node_cj)
    // where ainv_col_ci_d = A_d^{-1} * (n_ci[d] * e_{node_ci}).
    //
    // O(K·N) per-column sparse solves (K = num_active, N = vertex_num) instead
    // of the O(N²) ComputeSparseInverse; for K ≪ N this is orders of magnitude
    // faster. The STS cache is no longer populated or used.
    if (auto mesh_boundary = std::dynamic_pointer_cast<MeshFrictionalBoundary<vertex_dim>>(frictional_boundary_)) {
        mesh_boundary->UpdateVertices(q);
    }
    for (int ci = 0; ci < num_active; ++ci) {
        const int node = cache.contact_candidates[ci];
        const Eigen::Matrix<real, vertex_dim, 1> xj =
            q.segment(node * vertex_dim, vertex_dim);
        cache.normals[ci] = frictional_boundary_->GetNormal(xj, node);
    }

    if (num_active > 0) {
        // Ainv_cols_bwd[d][ci] = A_d^{-1} * (normals[ci][d] * e_{node_ci})
        std::vector<std::vector<Eigen::VectorXd>> ainv_cols_bwd(vertex_dim,
            std::vector<Eigen::VectorXd>(num_active));
        for (int d = 0; d < vertex_dim; ++d) {
            for (int ci = 0; ci < num_active; ++ci) {
                const int ni = cache.contact_candidates[ci];
                Eigen::VectorXd e_ci = Eigen::VectorXd::Zero(vertex_num);
                e_ci(ni) = static_cast<double>(cache.normals[ci](d));
                ainv_cols_bwd[d][ci] = apply_ainv(d, e_ci);
            }
        }
        for (int ci = 0; ci < num_active; ++ci) {
            for (int cj = ci; cj < num_active; ++cj) {
                const int nj = cache.contact_candidates[cj];
                double wij = 0.0;
                for (int d = 0; d < vertex_dim; ++d)
                    wij += static_cast<double>(cache.normals[ci](d))
                         * ainv_cols_bwd[d][ci](nj);
                cache.W(ci, cj) = wij;
                cache.W(cj, ci) = wij;
            }
        }
    }

    const std::map<int, real>& augmented_dirichlet = dirichlet_;
    VectorXr q_prev = q;

    // Anderson Acceleration + pd_rhs stability convergence (mirror forward).
    const int aa_window = options.count("aa_window")
                        ? static_cast<int>(options.at("aa_window")) : 5;
    std::deque<VectorXr> aa_q_hist, aa_g_hist;
    VectorXr prev_pd_rhs = VectorXr::Zero(dofs_);

    for (int iter = 0; iter < max_pd_iter; ++iter) {
        const VectorXr pd_rhs_vec =
            ProjectiveDynamicsLocalStep(cache.q_sol, a, augmented_dirichlet);

        // Must match forward: no ElasticForce in the PD RHS (see
        // deformable_projective_dynamics_forward_gpu.cpp) — with it the replay
        // diverges exactly as the forward pass did.
        cache.b_full = inv_h2m * cache.q_tilde + pd_rhs_vec;

        q_prev = cache.q_sol;

        if (auto mesh_boundary = std::dynamic_pointer_cast<MeshFrictionalBoundary<vertex_dim>>(frictional_boundary_)) {
            mesh_boundary->UpdateVertices(cache.q_sol);
        }
        for (int ci = 0; ci < num_active; ++ci) {
            const int node = cache.contact_candidates[ci];
            const Eigen::Matrix<real, vertex_dim, 1> xj =
                cache.q_sol.segment(node * vertex_dim, vertex_dim);
            cache.delta_n(ci) = static_cast<double>(frictional_boundary_->GetDistance(xj, node));
            cache.normals[ci] = frictional_boundary_->GetNormal(xj, node);
        }

        for (int ci = 0; ci < num_active; ++ci) {
            const double dj = cache.delta_n(ci);
            const double lj = cache.lambda(ci);
            const double rj = std::max(cache.W(ci, ci), 1e-10);

            double omega_j, E_j;
            double domega_ddelta, domega_dlambda;
            double dE_ddelta, dE_dlambda;

            alg_phd_exact_detail::ComputeOmegaAndE(
                dj, lj, rj,
                omega_j, E_j,
                domega_ddelta, domega_dlambda,
                dE_ddelta, dE_dlambda);

            cache.omega(ci)  = omega_j;
            cache.E_diag(ci) = E_j;
        }

        cache.M_sys.setZero();
        for (int ci = 0; ci < num_active; ++ci) {
            for (int cj = 0; cj < num_active; ++cj) {
                cache.M_sys(ci, cj) = cache.omega(ci) * cache.W(ci, cj) * cache.omega(cj);
            }
            cache.M_sys(ci, ci) += cache.E_diag(ci);
        }

        const Eigen::VectorXd omega_lambda = cache.omega.array() * cache.lambda.array();
        VectorXr JT_lambda = VectorXr::Zero(dofs_);
        for (int ci = 0; ci < num_active; ++ci) {
            const int node = cache.contact_candidates[ci];
            for (int d = 0; d < vertex_dim; ++d) {
                JT_lambda(node * vertex_dim + d) +=
                    static_cast<real>(cache.normals[ci](d) * omega_lambda(ci));
            }
        }
        cache.g_vec = cache.b_full + JT_lambda;

        Eigen::VectorXd h_vec(num_active);
        for (int ci = 0; ci < num_active; ++ci) {
            const double dj = cache.delta_n(ci);
            const double lj = cache.lambda(ci);
            const double rj = std::max(cache.W(ci, ci), 1e-10);

            double phi_j, dphi_ddelta, dphi_dlambda;
            alg_phd_exact_detail::ComputePhiAndDerivative(
                dj, lj, rj, phi_j, dphi_ddelta, dphi_dlambda);

            const int node = cache.contact_candidates[ci];
            double H_q = 0.0;
            for (int d = 0; d < vertex_dim; ++d) {
                H_q += static_cast<double>(cache.normals[ci](d))
                     * static_cast<double>(cache.q_sol(node * vertex_dim + d));
            }
            h_vec(ci) = -phi_j + cache.omega(ci) * H_q;
        }

        // A^{-1}·g_vec by sparse Cholesky solve, O(N), instead of the dense STS
        // product, O(N²). STS may be uninitialised when num_active == 0, so the
        // solver is always used.
        cache.Ainv_g.setZero();
        {
            const Eigen::Matrix<real, vertex_dim, -1> g_reshape =
                Eigen::Map<const Eigen::Matrix<real, vertex_dim, -1>>(
                    cache.g_vec.data(), vertex_dim, vertex_num);

            for (int d = 0; d < vertex_dim; ++d) {
                Eigen::VectorXd g_d = g_reshape.row(d).template cast<double>();
                const Eigen::VectorXd Ainv_gd = apply_ainv(d, g_d);
                for (int ni = 0; ni < vertex_num; ++ni) {
                    cache.Ainv_g(ni * vertex_dim + d) = static_cast<real>(Ainv_gd(ni));
                }
            }
        }

        Eigen::VectorXd J_Ainv_g(num_active);
        for (int ci = 0; ci < num_active; ++ci) {
            const int node = cache.contact_candidates[ci];
            double val = 0.0;
            for (int d = 0; d < vertex_dim; ++d) {
                val += static_cast<double>(cache.normals[ci](d))
                     * static_cast<double>(cache.Ainv_g(node * vertex_dim + d));
            }
            J_Ainv_g(ci) = cache.omega(ci) * val;
        }

        const Eigen::VectorXd rhs = h_vec - J_Ainv_g;
        cache.lambda += cache.M_sys.ldlt().solve(rhs);
        cache.lambda = cache.lambda.cwiseMax(0.0);

        const Eigen::VectorXd omega_lambda_new =
            cache.omega.array() * cache.lambda.array();
        VectorXr JT_lambda_new = VectorXr::Zero(dofs_);
        for (int ci = 0; ci < num_active; ++ci) {
            const int node = cache.contact_candidates[ci];
            for (int d = 0; d < vertex_dim; ++d) {
                JT_lambda_new(node * vertex_dim + d) +=
                    static_cast<real>(cache.normals[ci](d) * omega_lambda_new(ci));
            }
        }

        // Apply A^{-1} to b_corrected using sparse Cholesky solve (O(N)).
        const VectorXr b_corrected = cache.b_full + JT_lambda_new;
        const Eigen::Matrix<real, vertex_dim, -1> b_reshape =
            Eigen::Map<const Eigen::Matrix<real, vertex_dim, -1>>(
                b_corrected.data(), vertex_dim, vertex_num);

        for (int d = 0; d < vertex_dim; ++d) {
            Eigen::VectorXd b_d = b_reshape.row(d).template cast<double>();
            const Eigen::VectorXd q_d = apply_ainv(d, b_d);
            for (int ni = 0; ni < vertex_num; ++ni) {
                cache.q_sol(ni * vertex_dim + d) = static_cast<real>(q_d(ni));
            }
        }

        for (const auto& pair : augmented_dirichlet) {
            cache.q_sol(pair.first) = pair.second;
        }

        // ── Anderson Acceleration on cache.q_sol ─────────────────────────────
        // Mirrors the forward AA: push (q_prev, g_k = q_sol − q_prev) onto the
        // history deque, then from ≥ 2 entries solve the LS problem for q_AA.
        if (aa_window > 0) {
            VectorXr g_k = cache.q_sol - q_prev;
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
                const double reg = 1e-10 * std::max(GtG.trace() / std::max(m, 1), 1.0);
                GtG.diagonal().array() += reg;
                Eigen::VectorXd gamma = GtG.ldlt().solve(
                    dG.transpose() * g_k.template cast<double>());
                Eigen::VectorXd q_aa  = (q_prev + g_k).template cast<double>()
                                     - (dQ + dG) * gamma;
                cache.q_sol = q_aa.template cast<real>();
                for (const auto& pair : augmented_dirichlet)
                    cache.q_sol(pair.first) = pair.second;
            }
            while (static_cast<int>(aa_q_hist.size()) > aa_window) {
                aa_q_hist.pop_front();
                aa_g_hist.pop_front();
            }
        }

        // ── Convergence check: ||Δ q|| AND ||Δ pd_rhs|| ───────────────────────
        const real abs_err = (cache.q_sol - q_prev).norm();
        const real ref_err = q_prev.norm();
        const real pd_rhs_change = (pd_rhs_vec - prev_pd_rhs).norm();
        const real pd_rhs_ref    = prev_pd_rhs.norm();
        prev_pd_rhs = pd_rhs_vec;
        const bool q_step_small = (abs_err       <= rel_tol * ref_err    + abs_tol);
        const bool pd_rhs_small = (pd_rhs_change <= rel_tol * pd_rhs_ref + abs_tol);
        if (iter > 0 && q_step_small && pd_rhs_small) break;
    }
}

template<int vertex_dim, int element_dim>
VectorXr Deformable<vertex_dim, element_dim>::ApplyAlgPhd_Ainv(
        const AlgPhdReplayCache& cache,
        const VectorXr& rhs) const {
    VectorXr out = VectorXr::Zero(dofs_);
    const int vertex_num = mesh_.NumOfVertices();

    const Eigen::Matrix<real, vertex_dim, -1> rhs_reshape =
        Eigen::Map<const Eigen::Matrix<real, vertex_dim, -1>>(
            rhs.data(), vertex_dim, vertex_num);

    Eigen::Matrix<real, vertex_dim, -1> out_reshape(vertex_dim, vertex_num);

    // Prefer GPU SpMV cache (populated by forward); fall back to CPU LDLT
    // when the cache is not ready for this (d, vertex_num).
    for (int d = 0; d < vertex_dim; ++d) {
        Eigen::VectorXd rhs_d = rhs_reshape.row(d).template cast<double>();
        Eigen::VectorXd ainv_d(vertex_num);
        bool gpu_ok = false;
#ifdef CUDA_AVAILABLE
        gpu_ok = AlgPhd_GpuApplyAinv(d, vertex_num, rhs_d.data(), ainv_d.data());
#endif
        if (!gpu_ok) ainv_d = pd_eigen_solver_[d].solve(rhs_d);
        out_reshape.row(d) = ainv_d.template cast<real>();
    }

    out = Eigen::Map<const VectorXr>(out_reshape.data(), dofs_);
    return out;
}

template<int vertex_dim, int element_dim>
Eigen::VectorXd Deformable<vertex_dim, element_dim>::ApplyAlgPhd_J(
        const AlgPhdReplayCache& cache,
        const VectorXr& x) const {
    const int num_active = static_cast<int>(cache.contact_candidates.size());
    Eigen::VectorXd out = Eigen::VectorXd::Zero(num_active);

    for (int ci = 0; ci < num_active; ++ci) {
        const int node = cache.contact_candidates[ci];
        double val = 0.0;
        for (int d = 0; d < vertex_dim; ++d) {
            val += static_cast<double>(cache.normals[ci](d))
                 * static_cast<double>(x(node * vertex_dim + d));
        }
        out(ci) = cache.omega(ci) * val;
    }
    return out;
}

template<int vertex_dim, int element_dim>
VectorXr Deformable<vertex_dim, element_dim>::ApplyAlgPhd_JT(
        const AlgPhdReplayCache& cache,
        const Eigen::VectorXd& y) const {
    VectorXr out = VectorXr::Zero(dofs_);
    const int num_active = static_cast<int>(cache.contact_candidates.size());

    for (int ci = 0; ci < num_active; ++ci) {
        const int node = cache.contact_candidates[ci];
        const real s = static_cast<real>(cache.omega(ci) * y(ci));
        for (int d = 0; d < vertex_dim; ++d) {
            out(node * vertex_dim + d) += cache.normals[ci](d) * s;
        }
    }
    return out;
}

// keep declarations satisfied, but do not use the unstable dense block route
template<int vertex_dim, int element_dim>
VectorXr Deformable<vertex_dim, element_dim>::ApplyAlgPhd_RqLambda(
        const AlgPhdReplayCache& cache,
        const Eigen::VectorXd& dlambda) const {
    return -ApplyAlgPhd_Ainv(cache, ApplyAlgPhd_JT(cache, dlambda));
}

template<int vertex_dim, int element_dim>
Eigen::VectorXd Deformable<vertex_dim, element_dim>::ApplyAlgPhd_RlambdaLambda(
        const AlgPhdReplayCache& cache,
        const Eigen::VectorXd& dlambda) const {
    return cache.M_sys * dlambda;
}

template<int vertex_dim, int element_dim>
VectorXr Deformable<vertex_dim, element_dim>::ApplyAlgPhd_RqQ(
        const AlgPhdReplayCache& cache,
        const VectorXr& a,
        const std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                        vertex_dim * element_dim>>& pd_bwd_elem,
        const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                                    vertex_dim * element_dim>>>& pd_bwd_musc,
        const VectorXr& dq) const {
    const VectorXr db =
        ApplyProjectiveDynamicsLocalStepDifferential(
            cache.q_sol, a, pd_bwd_elem, pd_bwd_musc, dq)
        + ElasticForceDifferential(cache.q_sol, dq);

    return dq - ApplyAlgPhd_Ainv(cache, db);
}

template<int vertex_dim, int element_dim>
Eigen::VectorXd Deformable<vertex_dim, element_dim>::ApplyAlgPhd_RlambdaQ(
        const AlgPhdReplayCache& cache,
        const VectorXr& a,
        const std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                        vertex_dim * element_dim>>& pd_bwd_elem,
        const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                                    vertex_dim * element_dim>>>& pd_bwd_musc,
        const VectorXr& dq) const {
    const VectorXr db =
        ApplyProjectiveDynamicsLocalStepDifferential(
            cache.q_sol, a, pd_bwd_elem, pd_bwd_musc, dq)
        + ElasticForceDifferential(cache.q_sol, dq);

    return ApplyAlgPhd_J(cache, dq) - ApplyAlgPhd_J(cache, ApplyAlgPhd_Ainv(cache, db));
}

template<int vertex_dim, int element_dim>
VectorXr Deformable<vertex_dim, element_dim>::ApplyAlgPhd_RqQ_T(
    const AlgPhdReplayCache& cache,
    const VectorXr& y_q,
    const VectorXr& a,
    const std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                    vertex_dim * element_dim>>& pd_bwd_elem,
    const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                                vertex_dim * element_dim>>>& pd_bwd_musc) const {

    // Must be the transpose of the CURRENT simplified ApplyAlgPhd_RqQ:
    //   R_qq[dq] = dq - A^{-1}( db )
    // where db = d(local step) + d(elastic)

    const VectorXr z = ApplyAlgPhd_Ainv(cache, y_q);

    VectorXr out = y_q;
    out -= ApplyProjectiveDynamicsLocalStepDifferential(
        cache.q_sol, a, pd_bwd_elem, pd_bwd_musc, z);
    out -= ElasticForceDifferential(cache.q_sol, z);

    return out;
}

template<int vertex_dim, int element_dim>
VectorXr Deformable<vertex_dim, element_dim>::ApplyAlgPhd_RlambdaQ_T(
        const AlgPhdReplayCache& cache,
        const Eigen::VectorXd& y_lambda,
        const VectorXr& a,
        const std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                        vertex_dim * element_dim>>& pd_bwd_elem,
        const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                                    vertex_dim * element_dim>>>& pd_bwd_musc) const {

    // Must be the transpose of the CURRENT simplified ApplyAlgPhd_RlambdaQ:
    //   R_lambda,q[dq] = J dq - J A^{-1}(db)
    // where db = d(local step) + d(elastic)

    VectorXr out = ApplyAlgPhd_JT(cache, y_lambda);

    const VectorXr JT_y = ApplyAlgPhd_JT(cache, y_lambda);
    const VectorXr z    = ApplyAlgPhd_Ainv(cache, JT_y);

    out -= ApplyProjectiveDynamicsLocalStepDifferential(
        cache.q_sol, a, pd_bwd_elem, pd_bwd_musc, z);
    out -= ElasticForceDifferential(cache.q_sol, z);

    return out;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SolveAlgPhdFixedActiveAdjoint(
        const AlgPhdReplayCache& cache,
        const VectorXr& rhs_qbar,
        const VectorXr& a,
        const std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                        vertex_dim * element_dim>>& pd_bwd_elem,
        const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                                    vertex_dim * element_dim>>>& pd_bwd_musc,
        VectorXr& y_q,
        Eigen::VectorXd& y_lambda) const {

    const int nq = dofs_;
    const int nl = static_cast<int>(cache.contact_candidates.size());

    y_q = VectorXr::Zero(nq);
    y_lambda = Eigen::VectorXd::Zero(nl);

    Eigen::MatrixXd KT = Eigen::MatrixXd::Zero(nq + nl, nq + nl);

    // q-block columns: [Rqq^T e_j ; Rqlambda^T e_j]
    for (int j = 0; j < nq; ++j) {
        VectorXr e_q = VectorXr::Zero(nq);
        e_q(j) = 1.0;

        const VectorXr col_top = ApplyAlgPhd_RqQ_T(
            cache, e_q, a, pd_bwd_elem, pd_bwd_musc);

        Eigen::VectorXd col_bot = Eigen::VectorXd::Zero(nl);
        if (nl > 0) {
            // R_qlambda = -A^{-1} J^T
            // so R_qlambda^T e_q = -J A^{-1} e_q
            col_bot = -ApplyAlgPhd_J(cache, ApplyAlgPhd_Ainv(cache, e_q));
        }

        KT.block(0, j, nq, 1) = col_top;
        if (nl > 0) KT.block(nq, j, nl, 1) = col_bot;
    }

    // lambda-block columns: [Rlambdaq^T e_k ; Rlambdalambda^T e_k]
    for (int k = 0; k < nl; ++k) {
        Eigen::VectorXd e_l = Eigen::VectorXd::Zero(nl);
        e_l(k) = 1.0;

        const VectorXr col_top = ApplyAlgPhd_RlambdaQ_T(
            cache, e_l, a, pd_bwd_elem, pd_bwd_musc);

        const Eigen::VectorXd col_bot =
            ApplyAlgPhd_RlambdaLambda(cache, e_l);   // M_sys * e_l

        KT.block(0, nq + k, nq, 1) = col_top;
        KT.block(nq, nq + k, nl, 1) = col_bot;
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nq + nl);
    rhs.head(nq) = rhs_qbar;

    const Eigen::VectorXd sol = KT.fullPivLu().solve(rhs);

    y_q = sol.head(nq);
    if (nl > 0) y_lambda = sol.tail(nl);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamicsAlgPhd(
        const std::string& method,
        const VectorXr& q,
        const VectorXr& v,
        const VectorXr& a,
        const VectorXr& f_ext,
        const real dt,
        const VectorXr& q_next,
        const VectorXr& v_next,
        const std::vector<int>& active_contact_idx,
        const VectorXr& dl_dq_next,
        const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq,
        VectorXr& dl_dv,
        VectorXr& dl_da,
        VectorXr& dl_df_ext,
        VectorXr& dl_dmat_w,
        VectorXr& dl_dact_w,
        VectorXr& dl_dstate_p) const {

    CheckError(options.count("max_pd_iter"), "Missing option max_pd_iter.");
    CheckError(options.count("abs_tol"),     "Missing option abs_tol.");
    CheckError(options.count("rel_tol"),     "Missing option rel_tol.");
    CheckError(options.count("verbose"),     "Missing option verbose.");
    CheckError(options.count("thread_ct"),   "Missing option thread_ct.");

    const int thread_ct     = static_cast<int>(options.at("thread_ct"));
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const int max_pd_iter   = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol      = options.at("abs_tol");
    const real rel_tol      = options.at("rel_tol");

    omp_set_num_threads(thread_ct);

    const real h       = dt;
    const real inv_h   = ToReal(1) / h;
    const real mass    = element_volume_ * density_;
    const real h2m     = h * h / mass;
    const real inv_h2m = mass / (h * h);

    const std::string base_method = "pd_eigen";
    SetupProjectiveDynamicsSolver(base_method, dt, options);

    dl_dq       = VectorXr::Zero(dofs_);
    dl_dv       = VectorXr::Zero(dofs_);
    dl_da       = VectorXr::Zero(act_dofs_);
    dl_df_ext   = VectorXr::Zero(dofs_);
    dl_dmat_w   = VectorXr::Zero(NumOfPdElementEnergies());
    dl_dact_w   = VectorXr::Zero(NumOfPdMuscleEnergies());
    dl_dstate_p = VectorXr::Zero(NumOfStateForceParameters());

    // Backward through v_next = (q_next - q) / h
    dl_dq += -dl_dv_next * inv_h;
    const VectorXr rhs_qbar = dl_dq_next + dl_dv_next * inv_h;

    const bool use_precomputed_data = !pd_element_energies_.empty();
    if (use_precomputed_data) {
        ComputeDeformationGradientAuxiliaryDataAndProjection(q_next);
    }

    std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                              vertex_dim * element_dim>> pd_bwd_elem;
    std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                          vertex_dim * element_dim>>> pd_bwd_musc;
    SetupProjectiveDynamicsLocalStepDifferential(q_next, a, pd_bwd_elem, pd_bwd_musc);

    AlgPhdReplayCache replay;
    bool replay_from_tape = false;
    if (!alg_phd_forward_tape_.empty()) {
        const AlgPhdReplayCache& tail = alg_phd_forward_tape_.back();
        bool match = (tail.contact_candidates == active_contact_idx)
                  && (tail.q_sol.size() == q_next.size());
        if (match) {
            const real denom = std::max<real>(
                real(1), std::max(tail.q_sol.norm(), q_next.norm()));
            match = ((tail.q_sol - q_next).norm() <= real(1e-8) * denom);
        }
        if (match) {
            replay = tail;
            alg_phd_forward_tape_.pop_back();
            replay_from_tape = true;
        } else if (tail.q_sol.size() == q_next.size()) {
            // Stale/non-sequential tape: clear and fall back to replay to
            // avoid consuming mismatched frame caches.
            const real denom = std::max<real>(
                real(1), std::max(tail.q_sol.norm(), q_next.norm()));
            const real rel_err = (tail.q_sol - q_next).norm() / denom;
            if (rel_err > real(1e-6)) alg_phd_forward_tape_.clear();
        }
    }
    if (!replay_from_tape) {
        ReplayAlgPhdFixedActiveSet(q, v, a, f_ext, dt, active_contact_idx, options, replay);
    }

    if (verbose_level > 0 && !active_contact_idx.empty()) {
        std::cout << "[AlgPhd replay] src=" << (replay_from_tape ? "tape" : "replay")
                  << " active=" << replay.contact_candidates.size()
                  << "  |lambda|=" << replay.lambda.norm()
                  << "  |omega|=" << replay.omega.norm()
                  << "  |Ainv_g|=" << replay.Ainv_g.norm()
                  << "\n";
    }

    // =========================================================================
    // Backbone Adjoint: iterative fixed-point for (A - K_pd) μ = rhs_qbar
    //   db_grad ← A^{-1} · (rhs_qbar + K_pd · db_grad)
    //
    // Uses the CPU LDLT solver (pd_eigen_solver_[d]) directly, not
    // ApplyAlgPhd_Ainv: the latter routes through AlgPhd_GpuApplyAinv, which can
    // return NaN when the GPU SpMV state is inconsistent during backward. The
    // standard pd_eigen backward uses the same CPU LDLT. Dirichlet DOFs are
    // zeroed after each solve, matching pd_eigen backward.
    // =========================================================================
    const int vertex_num_bwd = mesh_.NumOfVertices();
    auto apply_ainv_cpu = [&](const VectorXr& rhs) -> VectorXr {
        Eigen::Map<const Eigen::Matrix<real, vertex_dim, -1>>
            rhs_mat(rhs.data(), vertex_dim, vertex_num_bwd);
        Eigen::Matrix<real, vertex_dim, -1> out_mat(vertex_dim, vertex_num_bwd);
        for (int d = 0; d < vertex_dim; ++d) {
            const Eigen::VectorXd rhs_d = rhs_mat.row(d).template cast<double>();
            out_mat.row(d) = pd_eigen_solver_[d].solve(rhs_d).template cast<real>();
        }
        VectorXr out = Eigen::Map<const VectorXr>(out_mat.data(), dofs_);
        for (const auto& pair : dirichlet_) out(pair.first) = ToReal(0);
        return out;
    };

    // Disabled GPU variant, kept for reference: A^{-1}·rhs = S^T·(S·rhs) via two
    // cuSPARSE ops on the factors forward left on the GPU, per-dimension CPU
    // SimplicialLDLT fallback, Dirichlet DOFs zeroed to prevent BC leakage.
//     auto apply_ainv = [&](const VectorXr& rhs) -> VectorXr {
//         Eigen::Map<const Eigen::Matrix<real, vertex_dim, -1>>
//             rhs_mat(rhs.data(), vertex_dim, vertex_num_bwd);
//         Eigen::Matrix<real, vertex_dim, -1> out_mat(vertex_dim, vertex_num_bwd);
//         for (int d = 0; d < vertex_dim; ++d) {
//             const Eigen::VectorXd rhs_d = rhs_mat.row(d).template cast<double>();
//             Eigen::VectorXd out_d(vertex_num_bwd);
//             bool gpu_ok = false;
// #ifdef CUDA_AVAILABLE
//             gpu_ok = AlgPhd_GpuApplyAinv(d, vertex_num_bwd, rhs_d.data(), out_d.data());
// #endif
//             if (!gpu_ok) out_d = pd_eigen_solver_[d].solve(rhs_d);
//             out_mat.row(d) = out_d.template cast<real>();
//         }
//         VectorXr out = Eigen::Map<const VectorXr>(out_mat.data(), dofs_);
//         for (const auto& pair : dirichlet_) out(pair.first) = ToReal(0);
//         return out;
//     };

    const int aa_window = options.count("aa_window")
                        ? static_cast<int>(options.at("aa_window")) : 5;
    std::deque<VectorXr> aa_x_hist, aa_g_hist;

    VectorXr db_grad = VectorXr::Zero(dofs_);
    int adjoint_iters = 0;
    const real db_clip_norm = options.count("alg_phd_bwd_clip_norm")
        ? options.at("alg_phd_bwd_clip_norm") : static_cast<real>(1e6);

    // Helper: apply NaN guard + norm clipping, return clipped vector.
    auto safe_clip = [&](VectorXr v) -> VectorXr {
        for (int j = 0; j < dofs_; ++j)
            if (!std::isfinite(v(j))) v(j) = real(0);
        const real n = v.norm();
        if (db_clip_norm > 0 && std::isfinite(n) && n > db_clip_norm)
            v *= (db_clip_norm / n);
        return v;
    };

    for (int i = 0; i < max_pd_iter; ++i) {
        const VectorXr kpd_term = ApplyProjectiveDynamicsLocalStepDifferential(
            q_next, a, pd_bwd_elem, pd_bwd_musc, db_grad);
        VectorXr db_grad_safe = safe_clip(apply_ainv_cpu(rhs_qbar + kpd_term));
        const real err = (db_grad_safe - db_grad).norm();

        // Anderson Acceleration (Type-II, Walker & Ni 2011) — same as forward.
        // Allows convergence when spectral radius of A^{-1}·K_pd ≥ 1 (plain
        // iteration diverges but AA can still converge via mixing).
        if (aa_window > 0) {
            const VectorXr g_k = db_grad_safe - db_grad;
            aa_x_hist.push_back(db_grad);
            aa_g_hist.push_back(g_k);
            const int m = static_cast<int>(aa_g_hist.size()) - 1;
            if (m >= 1) {
                Eigen::MatrixXd dG(dofs_, m), dQ(dofs_, m);
                for (int j = 0; j < m; ++j) {
                    dG.col(j) = (aa_g_hist[j + 1] - aa_g_hist[j]).template cast<double>();
                    dQ.col(j) = (aa_x_hist[j + 1] - aa_x_hist[j]).template cast<double>();
                }
                Eigen::MatrixXd GtG = dG.transpose() * dG;
                const double reg = std::max(1e-6 * dG.squaredNorm() / std::max(m, 1), 1e-12);
                GtG.diagonal().array() += reg;
                const Eigen::VectorXd gamma = GtG.ldlt().solve(
                    dG.transpose() * g_k.template cast<double>());
                if (gamma.norm() < 10.0) {
                    db_grad_safe = safe_clip(
                        ((db_grad + g_k).template cast<double>()
                         - (dQ + dG) * gamma).template cast<real>());
                } else {
                    aa_x_hist.clear();
                    aa_g_hist.clear();
                }
            }
            while (static_cast<int>(aa_x_hist.size()) > aa_window) {
                aa_x_hist.pop_front();
                aa_g_hist.pop_front();
            }
        }

        db_grad = db_grad_safe;
        adjoint_iters = i + 1;
        if (i > 0 && err < rel_tol * db_grad.norm() + abs_tol) break;
    }
    // backbone_diverged: adjoint hit the clip norm — the db_grad value is
    // unreliable.  Used to guard both cascade injection and Woodbury/mat_w.
    const bool backbone_diverged = (db_clip_norm > 0) &&
        (db_grad.norm() >= db_clip_norm * real(0.999));
    if (verbose_level > 0) {
        std::cout << "[AlgPhd-BWD] adjoint iters=" << adjoint_iters
                  << " db_grad.norm()=" << db_grad.norm()
                  << " rhs_qbar.norm()=" << rhs_qbar.norm()
                  << " diverged=" << backbone_diverged
                  << "\n";
    }

    // =========================================================================
    // 3. Downstream Gradient Routing
    // =========================================================================
    // Cascade guard: once the backbone adjoint has diverged (hit the clip), all
    // db_grad-derived contributions to dl_dq / dl_dv / dl_df_ext are skipped.
    // Otherwise one clipped frame injects inv_h2m * 1e6 into the previous
    // frame's rhs_qbar and every earlier backward frame hovers at ~1e6.
    if (!backbone_diverged) {
        // Relative cascade clipping: (A−K_pd)^{-1} ≈ h²/mass, so the expected
        // dl_dq injection is ≈ rhs_qbar.norm(). Contact frames with a
        // near-singular (A−K_pd) amplify that 100–1000×, and 21+ contact frames
        // in a 30-frame sim cascade exponentially even though no single frame
        // exceeds the hard 1e6 backbone clip.
        VectorXr dl_dq_tilde = inv_h2m * db_grad;
        {
            // 1× cap: each frame injects at most rhs_qbar.norm() into the
        // previous one. Well-conditioned frames inject ≈ rhs_qbar unclipped;
        // near-singular contact frames clip to rhs_qbar, so no amplification
        // accumulates across bounce events.
        const real cascade_cap = rhs_qbar.norm() + real(1.0);
            const real inject_norm = dl_dq_tilde.norm();
            if (inject_norm > cascade_cap)
                dl_dq_tilde *= (cascade_cap / inject_norm);
        }
        dl_dq     += dl_dq_tilde;
        dl_df_ext += h2m * dl_dq_tilde;

        // Velocity gradient uses the same (possibly clipped) injection vector.
        dl_dv     += h * dl_dq_tilde;

        // Through h2m * f_state(q, v)
        {
            VectorXr dl_dstate_q_contrib = VectorXr::Zero(dofs_);
            VectorXr dl_dstate_v_contrib = VectorXr::Zero(dofs_);
            const VectorXr f_state = ForwardStateForce(q, v);
            const VectorXr dl_df_state = (h2m * dl_dq_tilde).eval();

            BackwardStateForce(q, v, f_state, dl_df_state,
                               dl_dstate_q_contrib, dl_dstate_v_contrib, dl_dstate_p);

            dl_dq += dl_dstate_q_contrib;
            dl_dv += dl_dstate_v_contrib;
        }
    }

    // K_pd sensitivity is already absorbed into db_grad by the iterative solve above;
    // no separate local-step correction needed here.

    // =========================================================================
    // Contact-aware adjoint z_q (shared by dl/da and dl/d(mat_w))
    //   z_q = db_grad - W_T * G_T^{-1} * (J_basis * db_grad)
    //
    // Without this correction dl/da = db_grad^T * ∂f_act/∂a = 0 for any loss
    // with a uniform-translation gradient (e.g. COM displacement): db_grad =
    // (A−K_pd)^{-1} rhs_qbar ≈ (h²/m) rhs_qbar is a scaled translation field,
    // and internal (actuation) forces sum to zero against any rigid-body mode
    // by Newton's 3rd law.
    //
    // With active contacts the Woodbury term projects out the contact basis
    // directions (normal, plus tangents for 3D Coulomb friction), so z_q leaves
    // the translation field and dl/da becomes non-zero — capturing actuation →
    // deformation near contact → friction impulses → COM.
    // =========================================================================
    const bool skip_woodbury = options.count("skip_contact_woodbury")
                       && options.at("skip_contact_woodbury") > real(0.5);
    VectorXr z_q = db_grad;
    if (!backbone_diverged && !skip_woodbury) {
        const int nl = static_cast<int>(replay.contact_candidates.size());
        if (nl > 0) {
            const real omega_min = real(0.5);
            std::vector<int> active_ci;
            for (int ci = 0; ci < nl; ++ci)
                if (static_cast<real>(replay.omega(ci)) >= omega_min)
                    active_ci.push_back(ci);

            const real friction_mu_val = options.count("friction_mu")
                ? options.at("friction_mu") : real(0);
            const bool use_tangent_basis =
                (vertex_dim == 3) && (friction_mu_val > real(0));

            struct BasisConstraint {
                int ci;
                int basis;  // 0: normal, 1: t1, 2: t2
            };
            std::vector<Eigen::Matrix<real, vertex_dim, 1>> tangents1(nl);
            std::vector<Eigen::Matrix<real, vertex_dim, 1>> tangents2(nl);
            if constexpr (vertex_dim == 3) {
                if (use_tangent_basis) {
                    for (int ci = 0; ci < nl; ++ci) {
                        const Eigen::Matrix<real, 3, 1> n =
                            replay.normals[ci].template cast<real>();
                        int axis = 0;
                        for (int d = 1; d < 3; ++d)
                            if (std::abs(n(d)) < std::abs(n(axis))) axis = d;
                        Eigen::Matrix<real, 3, 1> e = Eigen::Matrix<real, 3, 1>::Zero();
                        e(axis) = real(1);
                        Eigen::Matrix<real, 3, 1> t1 = e - n.dot(e) * n;
                        const real t1n = t1.norm();
                        if (t1n > real(1e-12)) t1 /= t1n;
                        else t1 = Eigen::Matrix<real, 3, 1>(real(1), real(0), real(0));
                        Eigen::Matrix<real, 3, 1> t2 = n.cross(t1);
                        const real t2n = t2.norm();
                        if (t2n > real(1e-12)) t2 /= t2n;
                        else t2 = Eigen::Matrix<real, 3, 1>(real(0), real(1), real(0));
                        tangents1[ci] = t1;
                        tangents2[ci] = t2;
                    }
                }
            }

            std::vector<BasisConstraint> basis_constraints;
            basis_constraints.reserve(active_ci.size() * (use_tangent_basis ? 3 : 1));
            const real tangent_lambda_min = options.count("alg_phd_bwd_tangent_lambda_min")
                ? options.at("alg_phd_bwd_tangent_lambda_min") : real(1e-8);
            for (const int ci : active_ci) {
                basis_constraints.push_back({ci, 0});
                if (use_tangent_basis && replay.lambda(ci) > tangent_lambda_min) {
                    basis_constraints.push_back({ci, 1});
                    basis_constraints.push_back({ci, 2});
                }
            }

            const int nb_full = static_cast<int>(basis_constraints.size());
            if (nb_full > 0) {
                auto project_basis_value = [&](const BasisConstraint& bc,
                                               const VectorXr& x) -> double {
                    const int node = replay.contact_candidates[bc.ci];
                    if (bc.basis == 0) {
                        double val = 0.0;
                        for (int d = 0; d < vertex_dim; ++d) {
                            val += static_cast<double>(replay.normals[bc.ci](d))
                                 * static_cast<double>(x(node * vertex_dim + d));
                        }
                        return static_cast<double>(replay.omega(bc.ci)) * val;
                    }
                    if constexpr (vertex_dim == 3) {
                        const auto& t = (bc.basis == 1) ? tangents1[bc.ci] : tangents2[bc.ci];
                        double val = 0.0;
                        for (int d = 0; d < vertex_dim; ++d) {
                            val += static_cast<double>(t(d))
                                 * static_cast<double>(x(node * vertex_dim + d));
                        }
                        return val;
                    }
                    return 0.0;
                };

                Eigen::VectorXd rhs_full = Eigen::VectorXd::Zero(nb_full);
                double rhs_max_abs = 0.0;
                for (int i = 0; i < nb_full; ++i) {
                    rhs_full(i) = project_basis_value(basis_constraints[i], z_q);
                    rhs_max_abs = std::max(rhs_max_abs, std::abs(rhs_full(i)));
                }

                const real rhs_prune_rel = options.count("alg_phd_bwd_rhs_prune_rel")
                    ? options.at("alg_phd_bwd_rhs_prune_rel") : real(1e-3);
                const real rhs_prune_abs = options.count("alg_phd_bwd_rhs_prune_abs")
                    ? options.at("alg_phd_bwd_rhs_prune_abs") : real(1e-12);
                const int basis_topk = options.count("alg_phd_bwd_basis_topk")
                    ? static_cast<int>(options.at("alg_phd_bwd_basis_topk")) : 0;

                const double rhs_gate = std::max(
                    static_cast<double>(rhs_prune_abs),
                    static_cast<double>(rhs_prune_rel) * rhs_max_abs);

                std::vector<int> selected_idx;
                selected_idx.reserve(nb_full);
                for (int i = 0; i < nb_full; ++i)
                    if (std::abs(rhs_full(i)) >= rhs_gate) selected_idx.push_back(i);

                if (basis_topk > 0 && static_cast<int>(selected_idx.size()) > basis_topk) {
                    std::nth_element(
                        selected_idx.begin(),
                        selected_idx.begin() + basis_topk,
                        selected_idx.end(),
                        [&](int a, int b) {
                            return std::abs(rhs_full(a)) > std::abs(rhs_full(b));
                        });
                    selected_idx.resize(basis_topk);
                }

                const int nb = static_cast<int>(selected_idx.size());
                if (nb > 0) {
                    std::vector<BasisConstraint> selected_basis;
                    selected_basis.reserve(nb);
                    Eigen::VectorXd rhs_basis = Eigen::VectorXd::Zero(nb);
                    for (int i = 0; i < nb; ++i) {
                        selected_basis.push_back(basis_constraints[selected_idx[i]]);
                        rhs_basis(i) = rhs_full(selected_idx[i]);
                    }

                    auto apply_basis = [&](const VectorXr& x) -> Eigen::VectorXd {
                        Eigen::VectorXd out = Eigen::VectorXd::Zero(nb);
                        for (int i = 0; i < nb; ++i) {
                            out(i) = project_basis_value(selected_basis[i], x);
                        }
                        return out;
                    };

                    const int w_t_iter_cap = options.count("alg_phd_bwd_wt_max_iter")
                        ? static_cast<int>(options.at("alg_phd_bwd_wt_max_iter")) : 120;
                    const int w_t_max_iter = std::min(max_pd_iter, std::max(1, w_t_iter_cap));
                    std::vector<VectorXr> W_T(nb);
                    for (int k = 0; k < nb; ++k) {
                        const BasisConstraint bc = selected_basis[k];
                    const int ci = bc.ci;
                    VectorXr j_ci = VectorXr::Zero(dofs_);
                    const int node = replay.contact_candidates[ci];
                    if (bc.basis == 0) {
                        const real sc = static_cast<real>(replay.omega(ci));
                        for (int d = 0; d < vertex_dim; ++d)
                            j_ci(node * vertex_dim + d) = sc * replay.normals[ci](d);
                    } else if constexpr (vertex_dim == 3) {
                        const auto& t = (bc.basis == 1) ? tangents1[ci] : tangents2[ci];
                        for (int d = 0; d < vertex_dim; ++d)
                            j_ci(node * vertex_dim + d) = t(d);
                    }

                    VectorXr s = VectorXr::Zero(dofs_);
                    std::deque<VectorXr> wt_aa_x, wt_aa_g;
                    int w_t_iters = 0;
                    for (int it = 0; it < w_t_max_iter; ++it) {
                        const VectorXr kpd = ApplyProjectiveDynamicsLocalStepDifferential(
                                                 q_next, a, pd_bwd_elem, pd_bwd_musc, s);
                        VectorXr s_new = safe_clip(apply_ainv_cpu(j_ci + kpd));
                        const real err = (s_new - s).norm();

                        if (aa_window > 0) {
                            const VectorXr g_k = s_new - s;
                            wt_aa_x.push_back(s);
                            wt_aa_g.push_back(g_k);
                            const int m = static_cast<int>(wt_aa_g.size()) - 1;
                            if (m >= 1) {
                                Eigen::MatrixXd dG(dofs_, m), dQ(dofs_, m);
                                for (int j = 0; j < m; ++j) {
                                    dG.col(j) = (wt_aa_g[j+1] - wt_aa_g[j]).template cast<double>();
                                    dQ.col(j) = (wt_aa_x[j+1] - wt_aa_x[j]).template cast<double>();
                                }
                                Eigen::MatrixXd GtG = dG.transpose() * dG;
                                const double reg = std::max(1e-6 * dG.squaredNorm() / std::max(m, 1), 1e-12);
                                GtG.diagonal().array() += reg;
                                const Eigen::VectorXd gamma = GtG.ldlt().solve(
                                    dG.transpose() * g_k.template cast<double>());
                                if (gamma.norm() < 10.0) {
                                    s_new = safe_clip(
                                        ((s + g_k).template cast<double>()
                                         - (dQ + dG) * gamma).template cast<real>());
                                } else {
                                    wt_aa_x.clear();
                                    wt_aa_g.clear();
                                }
                            }
                            while (static_cast<int>(wt_aa_x.size()) > aa_window) {
                                wt_aa_x.pop_front();
                                wt_aa_g.pop_front();
                            }
                        }

                        s = s_new;
                        w_t_iters = it + 1;
                        if (it > 0 && err < rel_tol * s.norm() + abs_tol) break;
                    }
                    W_T[k] = s;
                    if (verbose_level > 0) {
                        std::cout << "[Woodbury] W_T[" << k << "] iters=" << w_t_iters
                                  << " ||j_ci||=" << j_ci.norm()
                                  << " ||W_T[k]||=" << s.norm() << "\n";
                    }
                }

                Eigen::MatrixXd G_T = Eigen::MatrixXd::Zero(nb, nb);
                for (int k = 0; k < nb; ++k) {
                    const Eigen::VectorXd jw = apply_basis(W_T[k]);
                    for (int l = 0; l < nb; ++l)
                        G_T(l, k) = jw(l);
                }

                if (verbose_level > 0) {
                    double rhs_n_max = 0.0, rhs_t_max = 0.0;
                    double rhs_n_l2 = 0.0, rhs_t_l2 = 0.0;
                    int rhs_n_nz = 0, rhs_t_nz = 0;
                    const double rhs_eps = 1e-14;
                    for (int i = 0; i < nb; ++i) {
                        const double v = std::abs(rhs_basis(i));
                        const bool is_normal = (selected_basis[i].basis == 0);
                        if (is_normal) {
                            rhs_n_max = std::max(rhs_n_max, v);
                            rhs_n_l2 += rhs_basis(i) * rhs_basis(i);
                            if (v > rhs_eps) ++rhs_n_nz;
                        } else {
                            rhs_t_max = std::max(rhs_t_max, v);
                            rhs_t_l2 += rhs_basis(i) * rhs_basis(i);
                            if (v > rhs_eps) ++rhs_t_nz;
                        }
                    }
                    std::cout << "[Woodbury] nl_act=" << static_cast<int>(active_ci.size())
                              << " basis_dim=" << nb
                              << " basis_dim_full=" << nb_full
                              << " ||db_grad||=" << db_grad.norm()
                              << " G_T(0,0)=" << G_T(0, 0)
                              << " rhs0=" << rhs_basis(0)
                              << " rhs|max|_n=" << rhs_n_max
                              << " rhs|max|_t=" << rhs_t_max
                              << " rhs|l2|_n=" << std::sqrt(rhs_n_l2)
                              << " rhs|l2|_t=" << std::sqrt(rhs_t_l2)
                              << " rhs_nz_n=" << rhs_n_nz
                              << " rhs_nz_t=" << rhs_t_nz
                              << " rhs_gate=" << rhs_gate
                              << "\n";
                }

                const Eigen::VectorXd alpha = G_T.ldlt().solve(rhs_basis);
                const real z_q_before = z_q.norm();
                for (int k = 0; k < nb; ++k)
                    z_q -= static_cast<real>(alpha(k)) * W_T[k];

                z_q = safe_clip(z_q);

                if (verbose_level > 0) {
                    std::cout << "[Woodbury] ||alpha||=" << alpha.norm()
                              << " ||z_q_before||=" << z_q_before
                              << " ||z_q_after||=" << z_q.norm()
                              << " ratio=" << z_q.norm() / (z_q_before + real(1e-30)) << "\n";
                }
                } else if (verbose_level > 0) {
                    std::cout << "[Woodbury] skipped (all rhs below gate)"
                              << " nl_act=" << static_cast<int>(active_ci.size())
                              << " basis_dim_full=" << nb_full
                              << " rhs_max_abs=" << rhs_max_abs
                              << " rhs_gate=" << rhs_gate
                              << "\n";
                }
            }
        }
    }

    // dl/da and dl/dact_w use the contact-aware z_q, not raw db_grad: db_grad is
    // a uniform translation field for COM-type losses (Newton's 3rd → zero dot
    // product), while z_q carries the contact corrections that make dl/da
    // non-zero under friction.
    if (!backbone_diverged) {
        SparseMatrixElements nonzeros_q_da, nonzeros_a, nonzeros_act_w_da;
        ActuationForceDifferential(q_next, a, nonzeros_q_da, nonzeros_a, nonzeros_act_w_da);
        const int act_w_dofs_da = NumOfPdMuscleEnergies();
        const VectorXr dl_da_contrib =
            VectorSparseMatrixProduct(z_q, dofs_, act_dofs_, nonzeros_a);
        const VectorXr dl_dact_w_contrib =
            VectorSparseMatrixProduct(z_q, dofs_, act_w_dofs_da, nonzeros_act_w_da);
        bool da_ok = true;
        for (int j = 0; j < dl_da_contrib.size() && da_ok; ++j)
            if (!std::isfinite(dl_da_contrib(j))) da_ok = false;
        if (da_ok) {
            dl_da    += dl_da_contrib;
            dl_dact_w += dl_dact_w_contrib;
        }
    }

    // dl/d(mat_w): reuse already-computed z_q (Woodbury correction shared above).
    {
        const bool skip_mat_grad = options.count("skip_material_grad")
                           && options.at("skip_material_grad") > real(0.5);
        const int mat_w_dofs = skip_mat_grad ? 0 : NumOfPdElementEnergies();
        if (mat_w_dofs > 0) {
            SparseMatrixElements nonzeros_q_dummy, nonzeros_mat_w;
            PdEnergyForceDifferential(q_next, false, true, use_precomputed_data,
                                      nonzeros_q_dummy, nonzeros_mat_w);

            const VectorXr frame_contrib = VectorSparseMatrixProduct(z_q, dofs_, mat_w_dofs, nonzeros_mat_w);
            bool contrib_ok = !backbone_diverged;
            if (contrib_ok) {
                for (int j = 0; j < frame_contrib.size(); ++j)
                    if (!std::isfinite(frame_contrib(j))) { contrib_ok = false; break; }
            }
            if (contrib_ok) dl_dmat_w += frame_contrib;
        }
    }
}

// =============================================================================
// Explicit template instantiations
// =============================================================================
template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;

// =============================================================================
// Helper: BackwardProjectiveDynamicsLocalStepDa
//   Backpropagates through the PdMuscleEnergy local step w.r.t. actuation a.
//   dl_da(act_idx + ei) += wi * (A^T M^T Ja)^T * mu_element
//   where Ja = ∂(ProjectToManifold) / ∂a.
// =============================================================================
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamicsLocalStepDa(
        const VectorXr& q_next,
        const VectorXr& a,
        const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                                          vertex_dim * element_dim>>>& pd_bwd_musc,
        const VectorXr& mu,
        VectorXr& dl_da) const {
    const int sample_num = GetNumOfSamplesInElement();
    int act_idx   = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy    = pair.first;
        const int element_cnt = static_cast<int>(pair.second.size());
        const real wi         = energy->stiffness() * element_volume_ / sample_num;
        #pragma omp parallel for
        for (int ei = 0; ei < element_cnt; ++ei) {
            const int i = pair.second[ei];
            const auto mu_elem = ScatterToElementFlattened(mu, i);
            real da_ei = 0;
            for (int j = 0; j < sample_num; ++j) {
                Eigen::Matrix<real, vertex_dim, vertex_dim * vertex_dim> JF;
                Eigen::Matrix<real, vertex_dim, 1> Ja;
                energy->ProjectToManifoldDifferential(
                    F_auxiliary_[i][j].F(), a(act_idx + ei), JF, Ja);
                const auto& Mt = energy->Mt();
                const Eigen::Matrix<real, vertex_dim * element_dim, 1> AtMtJa =
                    finite_element_samples_[i][j].pd_At() * Mt * Ja;
                da_ei += wi * AtMtJa.dot(mu_elem);
            }
            #pragma omp atomic
            dl_da(act_idx + ei) += da_ei;
        }
        act_idx   += element_cnt;
    }
}

// =============================================================================
// Helper: BackwardProjectiveDynamicsLocalStepDw
//   Backpropagates through the PdElementEnergy local step w.r.t. stiffness w.
//   dl_dmat_w(energy_idx) += (1/stiffness) * Σ_i Σ_j AtBpA_i^T * mu_i
//   (the 1/stiffness factor converts from d/d(w*stiffness) to d/d(stiffness))
// =============================================================================
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamicsLocalStepDw(
        const VectorXr& q_next,
        const std::vector<Eigen::Matrix<real, vertex_dim * element_dim,
                                              vertex_dim * element_dim>>& pd_bwd_elem,
        const VectorXr& mu,
        VectorXr& dl_dmat_w) const {
    const int sample_num  = GetNumOfSamplesInElement();
    const int element_num = mesh_.NumOfElements();
    int energy_idx = 0;
    for (const auto& epair : pd_element_energies_) {
        const auto& energy = epair.first;
        const auto& element_indices = epair.second;
        real dw = 0;
        for (int i = 0; i < element_num; ++i) {
            if (element_indices.find(i) == element_indices.end()) continue;
            const auto mu_elem = ScatterToElementFlattened(mu, i);
            const real scale = energy->element_stiffness(i) > 0
                ? (element_volume_ / sample_num)
                : 0;
            dw += scale * (pd_bwd_elem[i] * ScatterToElementFlattened(q_next, i)).dot(mu_elem);
        }
        if (energy_idx < static_cast<int>(dl_dmat_w.size()))
            dl_dmat_w(energy_idx) += dw;
        ++energy_idx;
    }
}
