#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDTetrahedronEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "Scomponent/tools/math/SVD.h"

namespace RealSim::integrator::projectivedynamics::energy{

// Implementation of tetrahedron energies with ARAP model
//
// Original Author: Ziqiu ZENG
//
class PDARAPTetrahedronEnergy : public PDTetrahedronEnergy
{

public:
    PDARAPTetrahedronEnergy(): PDTetrahedronEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: ARAP Tetrahedron Energy"<<RESET<<std::endl;
    }

protected:
    // solve nonlinear local problem
    Mat3R PDProjection(const Mat3R& F) override
    {
        Mat3R U, V, R;
        Vec3R sigma;

        RealSim::tools::svd::signedEigenSVD(F, U, sigma, V);

        R = U * V.transpose();

        return R;
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

            E += 0.5 * wi * (s - Vec3R::Identity()).squaredNorm();
        }

        return E;
    }
};


} //namespace sofa::localglobal::forcefield
