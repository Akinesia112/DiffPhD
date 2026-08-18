#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDHyperelasticTriangleEnergy.h"

namespace RealSim::integrator::projectivedynamics::energy{

// Implementation of triangle stretching energies with Neo-hookean elasticity
//
// Original Author: Ziqiu ZENG
//
class PDNeohookeanTriangleEnergy : public PDHyperelasticTriangleEnergy
{
public:
    typedef PDHyperelasticTriangleEnergy InheritEnergy;

    PDNeohookeanTriangleEnergy(): PDHyperelasticTriangleEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: Neo-Hookean Triangle Energy"<<RESET<<std::endl;
    }

    ProjectionProblem2D* get_problem() override { return &problem; }

    void init() override
    {
        InheritEnergy::init();
        problem.set_lame(this->_mu, this->_lambda);
    }

protected:
    NHProjectionProblem2D problem;
};


} //namespace sofa::localglobal::forcefield
