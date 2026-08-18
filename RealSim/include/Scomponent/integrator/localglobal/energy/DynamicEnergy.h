#pragma once
#include "Scomponent/integrator/localglobal/energy/BaseEnergy.h"

namespace RealSim::integrator::projectivedynamics::energy{


class DynamicEnergy: public BaseEnergy
{

public:
    DynamicEnergy(): BaseEnergy() {}
    DynamicEnergy(real weight): BaseEnergy(weight) {}

    // Assemble sum(w_i * D_i^T * D_i)
    void accumulateMatrix(MatXR &matrix) override
    {
        // we do nothing with accumulateMatrix in DynamicEnergy
    }

    // Assemble sum(w_i * D_i^T * D_i)
    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override
    {
        // we do nothing with accumulateMatrix in DynamicEnergy
    }

    virtual void accumulateDynamicMatrix(MatXR &matrix)
    {
        std::cout<<"DynamicEnergy::accumulateDynamicMatrix() not implemented"<<std::endl;
    }

    virtual void accumulateDynamicMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff)
    {
        std::cout<<"DynamicEnergy::accumulateDynamicMatrix() not implemented"<<std::endl;
    }
};

}
