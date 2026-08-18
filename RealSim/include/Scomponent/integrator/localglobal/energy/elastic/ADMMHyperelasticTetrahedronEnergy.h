#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/ADMMTetrahedronEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "Scomponent/tools/math/SVD.h"
#include "Scomponent/integrator/localglobal/energy/elastic/HyperelasticProblemS.h"

namespace RealSim::integrator::projectivedynamics::energy{

class ADMMHyperelasticTetrahedronEnergy : public ADMMTetrahedronEnergy
{

public:
    ADMMHyperelasticTetrahedronEnergy(): ADMMTetrahedronEnergy(){}

    mcl::optlib::LBFGS<double,3> solver;

    // Returns a pointer to the local problem (constitutive model)
    virtual ProjectionProblem3D* get_problem() {return nullptr;}

protected:
    // solve nonlinear local problem
    Mat3R ADMMProjection(const Mat3R& p) override
    {
        ProjectionProblem3D *problem = get_problem();

        Mat3R U, V, R;
        Vec3R s;

        RealSim::tools::svd::signedEigenSVD(p, U, s, V);

        Eigen::Vector3d sigma = s.cast<double>();
        Eigen::Vector3d sigma0 = sigma;

        // If everything is very low, It is collapsed to a point and the minimize
        // will likely fail. So we'll just inflate it a bit.
        const real eps = 1e-6;
        if (std::abs(sigma[0]) < eps && std::abs(sigma[1]) < eps && std::abs(sigma[2]) < eps) {
            sigma[0] = eps;
            sigma[1] = eps;
            sigma[2] = eps;
        }
        if( sigma[2] < 0.0 )
        {
            sigma[2] = -sigma[2];
        }

        solver.minimize(*problem, sigma, sigma0);

        R = U * (sigma.cast<real>()).asDiagonal() * V.transpose();

        return R;
    }
};


}
