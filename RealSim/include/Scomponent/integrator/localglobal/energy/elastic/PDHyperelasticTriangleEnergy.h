#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDTriangleStretchingEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "Scomponent/tools/math/SVD.h"
#include "Scomponent/integrator/localglobal/energy/elastic/HyperelasticProblemS.h"

namespace RealSim::integrator::projectivedynamics::energy{

// Base implementation of triangle stretching energies with hyperelastic elasticity
//
// Original Author: Ziqiu ZENG
//
class PDHyperelasticTriangleEnergy : public PDTriangleStretchingEnergy
{

public:
    PDHyperelasticTriangleEnergy(): PDTriangleStretchingEnergy(){}

    mcl::optlib::LBFGS<double,2> solver;

    // Returns a pointer to the local problem (constitutive model)
    virtual ProjectionProblem2D* get_problem() {return nullptr;}

protected:
    // solve nonlinear local problem
    Mat3x2R PDProjection(const Mat3x2R& F) override
    {
        ProjectionProblem2D *problem = get_problem();

        Mat3R U, V, R;

        Eigen::JacobiSVD<Mat3x2R > svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

        const Vec2R& s= svd.singularValues();
        Eigen::Vector2d sigma = s.cast<double>();
        Eigen::Vector2d sigma0 = sigma;

        // If everything is very low, It is collapsed to a point and the minimize
        // will likely fail. So we'll just inflate it a bit.
        const real eps = 1e-6;
        if (std::abs(sigma[0]) < eps && std::abs(sigma[1]) < eps) {
            sigma[0] = eps;
            sigma[1] = eps;
        }


        solver.minimize(*problem, sigma, sigma0);

        Mat3x2R P = svd.matrixU() * (sigma.cast<real>()).asDiagonal() * svd.matrixV().transpose();

        return P;
    }
};


}
