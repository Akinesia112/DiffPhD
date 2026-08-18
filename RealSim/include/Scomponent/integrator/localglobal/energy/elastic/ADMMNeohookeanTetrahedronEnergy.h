#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/ADMMHyperelasticTetrahedronEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"

namespace RealSim::integrator::projectivedynamics::energy{

class ADMMNeohookeanTetrahedronEnergy : public ADMMHyperelasticTetrahedronEnergy
{
public:
    typedef ADMMHyperelasticTetrahedronEnergy InheritEnergy;

    ADMMNeohookeanTetrahedronEnergy(): ADMMHyperelasticTetrahedronEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: Neo-Hookean Tetrahedron Energy (ADMM)"<<RESET<<std::endl;
    }

    ProjectionProblem3D* get_problem() override { return &problem; }

    void init() override
    {
        InheritEnergy::init();
        problem.set_lame(this->_mu, this->_lambda);
    }

protected:
    NHProjectionProblem3D problem;
};


} //namespace sofa::localglobal::forcefield
