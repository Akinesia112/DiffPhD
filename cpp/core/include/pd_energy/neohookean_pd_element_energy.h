#ifndef PD_ENERGY_NEOHOOKEAN_PD_ELEMENT_ENERGY_H
#define PD_ENERGY_NEOHOOKEAN_PD_ELEMENT_ENERGY_H

#include "pd_energy/pd_element_energy.h"

// Neo-Hookean energy as a PD element surrogate.
//
//   psi_NH(F) = (mu/2)*(||F||_F^2 - dim) - mu*ln(J) + (lambda/2)*(ln J)^2
//
// Per-element PD surrogate weight  w_e = 2*mu + lambda  (tangent modulus of
// psi_NH at F = I; paper/method_3.tex Eq. 9-11). The local step is the
// proximal projection on singular values:
//
//   sigma* = argmin_sigma  psi_NH(sigma) + (w_e/2) ||sigma - sigma_F||^2
//   p_e*   = U * diag(sigma*) * V^T
//
// RealSim reference: PDHyperelasticTetrahedronEnergy / HyperelasticProblemS.
template<int dim>
class NeoHookeanPdElementEnergy : public PdElementEnergy<dim> {
public:
    // Sets Lame parameters and the PD surrogate weight w_e = 2*mu + lambda.
    void InitializeNeoHookean(const real mu, const real lambda);
    // Per-element Lame. mu_ / lambda_ are stored as means (used by the proximal
    // projection's k); per-element w_e = 2*mu_e + lambda_e is registered via
    // InitializeElementStiffnesses so that the global A matrix is heterogeneous.
    void InitializeNeoHookeanPerElement(const std::vector<real>& mu_per_element,
                                        const std::vector<real>& lambda_per_element);

    const real mu()     const { return mu_; }
    const real lambda() const { return lambda_; }

    const Eigen::Matrix<real, dim, dim> ProjectToManifold(const Eigen::Matrix<real, dim, dim>& F) const override;
    const Eigen::Matrix<real, dim, dim> ProjectToManifoldDifferential(
        const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& dF) const override;

    const Eigen::Matrix<real, dim, dim> ProjectToManifold(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary) const override;
    const Eigen::Matrix<real, dim, dim> ProjectToManifoldDifferential(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection,
        const Eigen::Matrix<real, dim, dim>& dF) const override;

    // True NH first Piola–Kirchhoff stress and its differential. Overrides the
    // base-class PD-surrogate form `w_e*(F - p*)` with
    //    P(F) = mu*F - mu*F^{-T} + lambda*ln(J)*F^{-T}
    // so backward gradients match the underlying Neo-Hookean material rather
    // than the quadratic PD relaxation. Forward behaviour (ProjectToManifold
    // and the w_e·V·G^T·p* contribution to the global RHS) is unchanged.
    const real EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const override;
    const Eigen::Matrix<real, dim, dim> StressTensor(const Eigen::Matrix<real, dim, dim>& F) const override;
    const Eigen::Matrix<real, dim, dim> StressTensorDifferential(
        const Eigen::Matrix<real, dim, dim>& F,
        const Eigen::Matrix<real, dim, dim>& dF) const override;

    const real EnergyDensity(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
        const Eigen::Matrix<real, dim, dim>& projection) const override;
    const Eigen::Matrix<real, dim, dim> StressTensor(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
        const Eigen::Matrix<real, dim, dim>& projection) const override;
    const Eigen::Matrix<real, dim, dim> StressTensorDifferential(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
        const Eigen::Matrix<real, dim, dim>& projection,
        const Eigen::Matrix<real, dim, dim>& dF) const override;

private:
    const Eigen::Matrix<real, dim, 1> SolveProximal(const Eigen::Matrix<real, dim, 1>& sig_F) const;
    const Eigen::Matrix<real, dim, dim> HessianProximal(const Eigen::Matrix<real, dim, 1>& sig) const;

    real mu_     = 0;
    real lambda_ = 0;
};

#endif
