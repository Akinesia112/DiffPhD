#pragma once
#include <utility>

#include "Scomponent/integrator/localglobal/energy/BaseEnergy.h"

namespace RealSim::integrator::projectivedynamics::energy
{

// Implementation of static pin energy in Projective Dynamics
//
// Original Author: Ziqiu ZENG
//
class PinEnergy : public BaseEnergy
{

public:
    //user should give the full set of reference positions
    PinEnergy(real weight, VecXI vertices, MatX3R refpos)
        :BaseEnergy(weight), _vertices(std::move(vertices)), _refpos(std::move(refpos)){}

    // Assemble sum(w_i * D_i^T * D_i)
    void accumulateMatrix(MatXR &matrix) override;

    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override;

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    unsigned int getSize() override {return _vertices.size();}

    real getCurrentEnergy(const MatX3R &pos) override;

protected:
    VecXI _vertices;
    MatX3R _refpos;

};

}
