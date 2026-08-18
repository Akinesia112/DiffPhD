#pragma once
#include <utility>

#include "Scomponent/integrator/localglobal/energy/BaseEnergy.h"

namespace RealSim::integrator::projectivedynamics::energy
{

// Implementation of plane contact energies in PD
//
// Original Author: Ziqiu ZENG
//
class PlaneContactEnergy : public BaseEnergy
{
public:
    typedef BaseEnergy InheritEnergy;

    PlaneContactEnergy(real weight, const std::vector<int>& vertices, Vec3R origin, Vec3R normal)
        :BaseEnergy(weight), _vertices(vertices), _origin(std::move(origin)), _normal(std::move(normal)){}

    void init() override;

    // Assemble sum(w_i * D_i^T * D_i)
    void accumulateMatrix(MatXR &matrix) override;

    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override;

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

protected:
    std::vector<int> _vertices;     /// vertices
    Vec3R _origin;       ///< A point in the plane
    Vec3R _normal;       ///< The normal to the plane. Will be normalized by init().

};

} // namespace RealSim::integrator::localglobal::energy
