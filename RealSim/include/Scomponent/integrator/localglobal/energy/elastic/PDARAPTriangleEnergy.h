#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDTriangleStretchingEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "Scomponent/tools/math/SVD.h"

namespace RealSim::integrator::projectivedynamics::energy{

// Implementation of triangle stretching energies with ARAP model
//
// Original Author: Ziqiu ZENG
//
class PDARAPTriangleEnergy : public PDTriangleStretchingEnergy
{

public:
    PDARAPTriangleEnergy(): PDTriangleStretchingEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: ARAP Triangle Energy"<<RESET<<std::endl;
    }

protected:
    // solve nonlinear local problem
    Mat3x2R PDProjection(const Mat3x2R& F) override
    {
        Eigen::JacobiSVD<Mat3x2R > svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat3x2R S = Mat3x2R::Zero();
        S.block<2,2>(0,0) = Mat2R::Identity();
        Mat3x2R P = svd.matrixU() * S * svd.matrixV().transpose();

        return P;
    }
};


} //namespace sofa::localglobal::forcefield
