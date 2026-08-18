#pragma once
#include <utility>

#include "Scomponent/integrator/localglobal/energy/DynamicEnergy.h"
#include "Scomponent/integrator/localglobal/energy/contact/PlaneContactEnergy.h"

namespace RealSim::integrator::projectivedynamics::energy
{

class DynamicPlaneContactEnergy : public DynamicEnergy, public PlaneContactEnergy
{
public:
    typedef PlaneContactEnergy InheritEnergy;

    DynamicPlaneContactEnergy(real weight, const std::vector<int>& vertices, Vec3R origin, Vec3R normal)
        :DynamicEnergy(weight), PlaneContactEnergy(weight, vertices, origin, normal){}

    void init() override { InheritEnergy::init(); }

    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    void accumulateDynamicMatrix(MatXR &matrix) override;

    void accumulateDynamicMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override;

protected:
    std::vector<unsigned int> _verticesInContact;
};

} // namespace RealSim::integrator::localglobal::energy
