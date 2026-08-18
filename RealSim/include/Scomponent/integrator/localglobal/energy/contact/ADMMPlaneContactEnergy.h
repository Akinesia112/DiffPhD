#pragma once
#include <utility>

#include "Scomponent/integrator/localglobal/energy/contact/PlaneContactEnergy.h"

namespace RealSim::integrator::projectivedynamics::energy
{

// Implementation of plane contact energies in ADMM-PD (Overby et al. 2017)
//
// Original Author: Ziqiu ZENG
//
class ADMMPlaneContactEnergy : public PlaneContactEnergy
{
public:
    typedef PlaneContactEnergy InheritEnergy;

    ADMMPlaneContactEnergy(real weight, const std::vector<int>& vertices, Vec3R origin, Vec3R normal)
            :PlaneContactEnergy(weight, vertices, std::move(origin), std::move(normal))
    {
        // set size for pi, zi, ui
        _p = MatX3R::Zero(_vertices.size(), 3);
        _z = MatX3R::Zero(_vertices.size(), 3);
        _u = MatX3R::Zero(_vertices.size(), 3);
    }

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    void reset() override { resetU(); }

protected:
    MatX3R _p, _z, _u;

    // solve nonlinear local problem
    void projectZ();

    void resetU();
};

} // namespace RealSim::integrator::localglobal::energy
