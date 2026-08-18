#pragma once
#include <utility>

#include "Scomponent/integrator/localglobal/energy/BaseEnergy.h"


namespace RealSim::integrator::projectivedynamics::energy
{

// Implementation of axis pin energy in Projective Dynamics (allows for rolling)
//
// Original Author: Ziqiu ZENG
//
class AxisPinEnergy : public BaseEnergy
{

public:
    AxisPinEnergy(real weight, MatX3R refpos, Vec3R base, Vec3R axis, real radius, bool slide)
        :BaseEnergy(weight), _refpos(std::move(refpos)), _base(std::move(base)), _axis(std::move(axis)), _radius(radius), _slide(slide)
        {}

    void init() override;

    // Assemble sum(w_i * D_i^T * D_i)
    void accumulateMatrix(MatXR &matrix) override;

    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override;

    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    unsigned int getSize() override {return _vertices.size();}

protected:
    std::vector<int> _vertices;
    std::vector<real> _r;
    MatX3R _refpos;

    Vec3R _base;
    Vec3R _axis;
    real _radius;

    bool _slide;
};

}
