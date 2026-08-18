#pragma once
#include "printconfig.h"
#include "types.h"

namespace RealSim::integrator::projectivedynamics::energy{

class BaseEnergy
{

public:
    BaseEnergy() {}
    BaseEnergy(real weight): _weight(weight) {}

    virtual void init() {}

    // Assemble sum(w_i * D_i^T * D_i)
    virtual void accumulateMatrix(MatXR &matrix)
    {
        std::cout<<"BaseEnergy::assembleGlobalMatrix() not implemented"<<std::endl;
    }

    // Assemble sum(w_i * D_i^T * D_i) using triplet set
    virtual void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff)
    {
        std::cout<<"BaseEnergy::assembleGlobalMatrix() not implemented"<<std::endl;
    }

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    virtual void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff)
    {
        std::cout<<"BaseEnergy::localProjection() not implemented"<<std::endl;
    }

    // reset operations in each time step
    virtual void reset(){}

    virtual real getCurrentEnergy(const MatX3R &pos)
    {
        std::cout<<"BaseEnergy::getCurrentEnergy() not implemented"<<std::endl;
        return 0.0;
    }

    virtual unsigned int getSize()
    {
        std::cout<<"BaseEnergy::getSize() not implemented"<<std::endl;
        return 0;
    }

protected:
    real _weight;
};

}
