#include "pd_energy/neohookean_pd_element_energy.h"
#include "common/common.h"
#include "common/geometry.h"
#include <numeric>

template<int dim>
void NeoHookeanPdElementEnergy<dim>::InitializeNeoHookean(const real mu, const real lambda) {
    mu_ = mu;
    lambda_ = lambda;
    // Consistent PD: A-matrix weight = proximal penalty k = 2μ (RealSim convention).
    // PD fixed-point Jacobian eigenvalue = k/(H_ψ+k) < 1 for any k>0.
    // Anderson Acceleration (aa_window=5) is required to converge at tight tol for high ν.
    this->Initialize(ToReal(2) * mu);
}

template<int dim>
void NeoHookeanPdElementEnergy<dim>::InitializeNeoHookeanPerElement(
    const std::vector<real>& mu_per_element,
    const std::vector<real>& lambda_per_element) {
    CheckError(mu_per_element.size() == lambda_per_element.size(),
               "mu/lambda per-element length mismatch.");
    const int n = static_cast<int>(mu_per_element.size());
    CheckError(n > 0, "Empty per-element lame arrays.");

    // Store means for the proximal projection (k in psi + (k/2)||sigma - sigma_F||^2).
    // v1: scalar k only; projection ignores per-element variation. The global A
    // matrix is correctly heterogeneous via element_stiffness(i) below.
    mu_     = std::accumulate(mu_per_element.begin(),     mu_per_element.end(),     real(0)) / n;
    lambda_ = std::accumulate(lambda_per_element.begin(), lambda_per_element.end(), real(0)) / n;

    std::vector<real> w_per(n);
    for (int i = 0; i < n; ++i)
        w_per[i] = ToReal(2) * mu_per_element[i];  // Consistent: k = 2μ
    this->InitializeElementStiffnesses(w_per);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::HessianProximal(
    const Eigen::Matrix<real, dim, 1>& sig) const {
    // Consistent: proximal penalty k = stiffness() = 2μ.
    const real k = this->stiffness();
    const real sig_prod = sig.prod();
    const real L = std::log(std::max(sig_prod, ToReal(1e-12)));
    Eigen::Matrix<real, dim, dim> H;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (i == j) {
                const real s2 = sig(i) * sig(i);
                H(i, j) = mu_ + (mu_ + lambda_ - lambda_ * L) / s2 + k;
            } else {
                H(i, j) = lambda_ / (sig(i) * sig(j));
            }
        }
    }
    return H;
}

template<int dim>
const Eigen::Matrix<real, dim, 1> NeoHookeanPdElementEnergy<dim>::SolveProximal(
    const Eigen::Matrix<real, dim, 1>& sig_F) const {
    // Consistent: proximal penalty k = stiffness() = 2μ.
    const real k = this->stiffness();
    const real sig_min = ToReal(1e-4);
    const real tol = ToReal(1e-10);
    const int max_iter = 32;

    Eigen::Matrix<real, dim, 1> sig = sig_F;
    for (int i = 0; i < dim; ++i)
        if (sig(i) < sig_min) sig(i) = sig_min;

    for (int iter = 0; iter < max_iter; ++iter) {
        const real sig_prod = sig.prod();
        const real L = std::log(std::max(sig_prod, ToReal(1e-12)));
        Eigen::Matrix<real, dim, 1> g;
        for (int i = 0; i < dim; ++i) {
            g(i) = mu_ * sig(i) - mu_ / sig(i) + lambda_ * L / sig(i)
                 + k * (sig(i) - sig_F(i));
        }
        if (g.norm() < tol) break;

        Eigen::Matrix<real, dim, dim> H = HessianProximal(sig);
        for (int i = 0; i < dim; ++i) H(i, i) += ToReal(1e-9);

        const Eigen::Matrix<real, dim, 1> delta = H.ldlt().solve(g);

        // Back-tracking line search with positivity guard.
        real alpha = ToReal(1);
        Eigen::Matrix<real, dim, 1> sig_new = sig - alpha * delta;
        for (int ls = 0; ls < 20; ++ls) {
            real min_s = sig_new(0);
            for (int i = 1; i < dim; ++i) if (sig_new(i) < min_s) min_s = sig_new(i);
            if (min_s > sig_min * ToReal(0.5)) break;
            alpha *= ToReal(0.5);
            sig_new = sig - alpha * delta;
        }

        const real step = (sig_new - sig).cwiseAbs().maxCoeff();
        for (int i = 0; i < dim; ++i)
            if (sig_new(i) < sig_min) sig_new(i) = sig_min;
        sig = sig_new;
        if (step < tol) break;
    }
    return sig;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::ProjectToManifold(
    const Eigen::Matrix<real, dim, dim>& F) const {
    DeformationGradientAuxiliaryData<dim> F_aux;
    F_aux.Initialize(F);
    return ProjectToManifold(F_aux);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::ProjectToManifold(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary) const {
    const auto& U   = F_auxiliary.U();
    const auto& V   = F_auxiliary.V();
    const auto& sig = F_auxiliary.sig();
    const auto sig_star = SolveProximal(sig);
    return U * sig_star.asDiagonal() * V.transpose();
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& dF) const {
    DeformationGradientAuxiliaryData<dim> F_aux;
    F_aux.Initialize(F);
    return ProjectToManifoldDifferential(F_aux, Eigen::Matrix<real, dim, dim>::Zero(), dF);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::ProjectToManifoldDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& /*projection*/,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    const auto& F   = F_auxiliary.F();
    const auto& U   = F_auxiliary.U();
    const auto& V   = F_auxiliary.V();
    const auto& sig = F_auxiliary.sig();

    // Forward proximal solution sigma*.
    const auto sig_star = SolveProximal(sig);
    // Consistent: proximal penalty k = stiffness() = 2μ.
    const real k = this->stiffness();

    // dSVD: dF -> (dU, dsig_F, dV).
    Eigen::Matrix<real, dim, 1> dsig_F;
    Eigen::Matrix<real, dim, dim> dU, dV;
    dSvd(F, U, sig, V, dF, dU, dsig_F, dV);

    // IFT on prox: (H_psi(sig*) + k*I) * dsig* = k * dsig_F.
    // HessianProximal already returns H_psi + k*I.
    Eigen::Matrix<real, dim, dim> H_prox = HessianProximal(sig_star);
    for (int i = 0; i < dim; ++i) H_prox(i, i) += ToReal(1e-9);
    const Eigen::Matrix<real, dim, 1> dsig_star = H_prox.ldlt().solve(k * dsig_F);

    // dp* = dU*diag(sig*)*V^T + U*diag(dsig*)*V^T + U*diag(sig*)*dV^T.
    return dU * sig_star.asDiagonal() * V.transpose()
         + U * dsig_star.asDiagonal() * V.transpose()
         + U * sig_star.asDiagonal() * dV.transpose();
}

// ──────────────────────────────────────────────────────────────────────────
// True Neo-Hookean PK1 stress and tangent (mirrors NeohookeanMaterial so that
// the backward adjoint sees the same P as the underlying elastic material,
// avoiding surrogate-vs-true drift that causes |grad| blow-up over many
// frames). Forward still uses ProjectToManifold for the PD global-step RHS.
// ──────────────────────────────────────────────────────────────────────────

template<int dim>
const real NeoHookeanPdElementEnergy<dim>::EnergyDensity(
    const Eigen::Matrix<real, dim, dim>& F) const {
    const real I1 = F.squaredNorm();
    const real J = F.determinant();
    const real log_j = std::log(std::max(J, ToReal(1e-12)));
    return mu_ * ToReal(0.5) * (I1 - dim) - mu_ * log_j
         + lambda_ * ToReal(0.5) * log_j * log_j;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::StressTensor(
    const Eigen::Matrix<real, dim, dim>& F) const {
    const real J = F.determinant();
    const real log_j = std::log(std::max(J, ToReal(1e-12)));
    const Eigen::Matrix<real, dim, dim> F_inv_T = F.inverse().transpose();
    return mu_ * (F - F_inv_T) + lambda_ * log_j * F_inv_T;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    const real J = F.determinant();
    const real log_j = std::log(std::max(J, ToReal(1e-12)));
    const Eigen::Matrix<real, dim, dim> F_inv   = F.inverse();
    const Eigen::Matrix<real, dim, dim> F_inv_T = F_inv.transpose();
    // d(F^-1)  = -F^-1 dF F^-1
    // d(F^-T) = -F^-T dF^T F^-T
    const Eigen::Matrix<real, dim, dim> dF_inv_T = -(F_inv_T * dF.transpose() * F_inv_T);
    const real dlog_j = (F_inv_T.array() * dF.array()).sum();
    return mu_ * (dF - dF_inv_T) + lambda_ * (log_j * dF_inv_T + dlog_j * F_inv_T);
}

template<int dim>
const real NeoHookeanPdElementEnergy<dim>::EnergyDensity(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& /*projection*/) const {
    return EnergyDensity(F_auxiliary.F());
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::StressTensor(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& /*projection*/) const {
    return StressTensor(F_auxiliary.F());
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeoHookeanPdElementEnergy<dim>::StressTensorDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& /*projection*/,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    return StressTensorDifferential(F_auxiliary.F(), dF);
}

template class NeoHookeanPdElementEnergy<2>;
template class NeoHookeanPdElementEnergy<3>;
