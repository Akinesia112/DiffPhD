#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDHyperelasticTetrahedronEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"

namespace RealSim::integrator::projectivedynamics::energy{

// Implementation of tetrahedron energies with Neo-hookean elasticity
//
// Original Author: Ziqiu ZENG
//
class PDNeohookeanTetrahedronEnergy : public PDHyperelasticTetrahedronEnergy
{
public:
    typedef PDHyperelasticTetrahedronEnergy InheritEnergy;

    PDNeohookeanTetrahedronEnergy(): PDHyperelasticTetrahedronEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: Neo-Hookean Tetrahedron Energy"<<RESET<<std::endl;
    }

    ProjectionProblem3D* get_problem() override { return &problem; }

    void init() override
    {
        InheritEnergy::init();
        problem.set_lame(this->_mu, this->_lambda);
    }

    real getCurrentEnergy(const MatX3R &pos) override
    {
        real E = 0.0;

        Mat3R F;
        real wi;

        for(unsigned int  i = 0 ; i<getElementDimension(); i++) {
            const Eigen::Vector4i &tetra = _geometry->getTetrahedra()->row(i);

            wi = this->_weight * _vol[i];

            computeDeformationGradient(F, pos, i);

            Mat3R U, V, R;
            Vec3R s;
            RealSim::tools::svd::signedEigenSVD(F, U, s, V);

            E += 0.5 * wi * problem.energy_density(s);
        }

        return E;
    }

protected:
    NHProjectionProblem3D problem;
};


} //namespace sofa::localglobal::forcefield
