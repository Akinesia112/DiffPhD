#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/ADMMTetrahedronEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "Scomponent/tools/math/SVD.h"

namespace RealSim::integrator::projectivedynamics::energy{

class ADMMARAPTetrahedronEnergy : public ADMMTetrahedronEnergy
{

public:
    ADMMARAPTetrahedronEnergy(): ADMMTetrahedronEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: ARAP Tetrahedron Energy (ADMM)"<<RESET<<std::endl;
    }

protected:
    // solve nonlinear local problem
    Mat3R ADMMProjection(const Mat3R& p) override
    {
        Mat3R U, V, R;
        Vec3R sigma;

        RealSim::tools::svd::signedEigenSVD(p, U, sigma, V);

        R = U * V.transpose();

        return 0.5 * (p + R);
    }
};


} //namespace sofa::localglobal::forcefield
