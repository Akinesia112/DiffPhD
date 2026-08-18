#include "pd_energy/pd_element_energy.h"
#include <numeric>

template<int dim>
void PdElementEnergy<dim>::Initialize(const real stiffness) {
    stiffness_ = stiffness;
    element_stiffnesses_.clear();
}

template<int dim>
void PdElementEnergy<dim>::InitializeElementStiffnesses(const std::vector<real>& stiffnesses) {
    element_stiffnesses_ = stiffnesses;
    // Keep stiffness_ as the mean so code paths that still read stiffness()
    // (e.g., the backward local-step correction) receive a sane scalar.
    stiffness_ = stiffnesses.empty() ? stiffness_
        : std::accumulate(stiffnesses.begin(), stiffnesses.end(), real(0)) / static_cast<real>(stiffnesses.size());
}

template<int dim>
const real PdElementEnergy<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const {
    return stiffness_ * 0.5 * (F - ProjectToManifold(F)).squaredNorm();
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdElementEnergy<dim>::StressTensor(const Eigen::Matrix<real, dim, dim>& F) const {
    return stiffness_ * (F - ProjectToManifold(F));
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdElementEnergy<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    return stiffness_ * (dF - ProjectToManifoldDifferential(F, dF));
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> PdElementEnergy<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim * dim, dim * dim> I;
    I.setZero();
    for (int i = 0; i < dim * dim; ++i) I(i, i) = 1;
    return stiffness_ * (I - ProjectToManifoldDifferential(F));
}

template<int dim>
const real PdElementEnergy<dim>::EnergyDensity(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& projection) const {
    return stiffness_ * 0.5 * (F_auxiliary.F() - projection).squaredNorm();
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdElementEnergy<dim>::StressTensor(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& projection) const {
    return stiffness_ * (F_auxiliary.F() - projection);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdElementEnergy<dim>::StressTensorDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& projection, const Eigen::Matrix<real, dim, dim>& dF) const {
    return stiffness_ * (dF - ProjectToManifoldDifferential(F_auxiliary, projection, dF));
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> PdElementEnergy<dim>::StressTensorDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection) const {
    Eigen::Matrix<real, dim * dim, dim * dim> I;
    I.setZero();
    for (int i = 0; i < dim * dim; ++i) I(i, i) = 1;
    return stiffness_ * (I - ProjectToManifoldDifferential(F_auxiliary, projection));
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> PdElementEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim * dim, dim * dim> J;
    J.setZero();
    for (int i = 0; i < dim * dim; ++i) {
        Eigen::Matrix<real, dim * dim, 1> dF;
        dF.setZero();
        dF(i) = 1;
        const Eigen::Matrix<real, dim, dim> F_col = ProjectToManifoldDifferential(F,
            Eigen::Map<const Eigen::Matrix<real, dim, dim>>(dF.data(), dim, dim));
        J.col(i) = Eigen::Map<const Eigen::Matrix<real, dim * dim, 1>>(F_col.data(), F_col.size());
    }
    return J;
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> PdElementEnergy<dim>::ProjectToManifoldDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection) const {
    Eigen::Matrix<real, dim * dim, dim * dim> J;
    J.setZero();
    for (int i = 0; i < dim * dim; ++i) {
        Eigen::Matrix<real, dim * dim, 1> dF;
        dF.setZero();
        dF(i) = 1;
        const Eigen::Matrix<real, dim, dim> F_col = ProjectToManifoldDifferential(F_auxiliary, projection,
            Eigen::Map<const Eigen::Matrix<real, dim, dim>>(dF.data(), dim, dim));
        J.col(i) = Eigen::Map<const Eigen::Matrix<real, dim * dim, 1>>(F_col.data(), F_col.size());
    }
    return J;
}

template class PdElementEnergy<2>;
template class PdElementEnergy<3>;