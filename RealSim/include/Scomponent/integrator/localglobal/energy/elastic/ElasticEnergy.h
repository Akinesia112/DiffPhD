#pragma once
#include "Scomponent/integrator/localglobal/energy/BaseEnergy.h"
#include "Sgeometry/SBaseGeometry.h"

namespace RealSim::integrator::projectivedynamics::energy
{

enum ElasticModel {
    TRI_ARAP,
    TRI_NEOHOOKEAN,
    TRI_COROTATION,
    TRI_SPRING,
    BEND_ISOMETRIC,
    BEND_LAPLACEBELTRAMI,
    TET_PD_ARAP,
    TET_PD_NEOHOOKEAN,
    TET_PD_COROTATION,
    TET_ADMM_ARAP,
    TET_ADMM_NEOHOOKEAN,
};

std::string model2String(ElasticModel m);

ElasticModel string2Model(const std::string& s);

class ElasticEnergy : public BaseEnergy
{

public:
    ElasticEnergy(): BaseEnergy(){}

    virtual void setGeometry(geometry::SBaseGeometry * geometry) = 0;

    virtual void setElasticParameter(real youngModulus, real poissonRatio)
    {
        _youngModulus = youngModulus;
        _poissonRatio = poissonRatio;

        // compute lame coefficients
        _mu = _youngModulus / (2.0 * (1.0 + _poissonRatio));
        _lambda = (_youngModulus * _poissonRatio) / ((1.0 + _poissonRatio) * (1.0 - 2.0 * _poissonRatio));

        _weight = 2 * _mu;
    }

    virtual unsigned int getVertexDimension() = 0;

    virtual unsigned int getElementDimension() = 0;

protected:
    real _youngModulus, _poissonRatio;
    real _mu, _lambda;

};


} //namespace sofa::localglobal::forcefield
