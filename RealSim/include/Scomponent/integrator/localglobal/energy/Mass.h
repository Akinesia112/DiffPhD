#pragma once
#include <iostream>
#include "types.h"

namespace RealSim::integrator::projectivedynamics::energy{

// Implementation of mass component in PD
//
// Original Author: Ziqiu ZENG
//
class Mass
{

public:
    Mass(unsigned int dim);

    void addObjectMass(real value, const VecXI & vertices);

    void addVertexMass(real value, const VecXI & vertices);

    void accumulateMatrix(MatXR &matrix);

    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff);

    // compute s_n = pos + h * vel + h^2 * M^-1 * f
    void computeSn(MatX3R &sn, const MatX3R &pos, const MatX3R &vel, const MatX3R &f, real dt);

    // res = M * x
    void computeMx(MatX3R &res, const MatX3R &x, real coeff);

    real getCurrentEnergy(const MatX3R &pos, const MatX3R &sn, real dt);

protected:
    unsigned int _dim;
    VecXR _mass, _invmass;
};

}
