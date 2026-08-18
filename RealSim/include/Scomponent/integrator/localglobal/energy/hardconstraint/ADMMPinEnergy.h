#pragma once
#include <utility>

#include "Scomponent/integrator/localglobal/energy/hardconstraint/PinEnergy.h"


namespace RealSim::integrator::projectivedynamics::energy
{

class ADMMPinEnergy : public PinEnergy
{

public:
    typedef PinEnergy InheritEnergy;

    ADMMPinEnergy(real weight, VecXI vertices, MatX3R refpos)
        :PinEnergy(weight, std::move(vertices), std::move(refpos)){}

    void init() override
    {
        InheritEnergy::init();

        _p.resize(this->getSize());
        _z.resize(this->getSize());
        _u.resize(this->getSize());
    }

    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override
    {
        auto weight = coeff * this->_weight;

        for(unsigned i=0; i<getSize(); i++)
        {
            unsigned int index = this->_vertices[i];
            Vec3R p = pos.row(index);

            _p[i] = p + _u[i];

            _z[i] = _refpos.row(index);

            _u[i] += (p - _z[i]);

            Vec3R res = _z[i] - _u[i];

            rhs.row(index) += weight * res;
        }
    }

    void reset() override
    {
        _u.clear();
        _u.resize(this->getSize(), Vec3R::Zero());
    }

protected:
    std::vector<Vec3R> _p, _z, _u;
};

}
