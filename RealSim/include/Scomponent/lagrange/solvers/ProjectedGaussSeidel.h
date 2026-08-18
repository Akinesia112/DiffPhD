#pragma once
#include "Scomponent/lagrange/solvers/LinearizedConstraintSolver.h"

namespace RealSim::lagrange::constraintsolver
{

// Implementation of Projected Gauss-Seidel for contact (LCP) and frictional contact (NCP) constraint resolution
//
// Original Author: Ziqiu ZENG
//
class PGSSolver : public LinearizedConstraintSolver
{
public:
    void solve() override;

protected:
    void BilateralResolution(int line, const MatXR & w, real* d, real* force, int size);
    void UnilateralResolution(int line, const MatXR & w, real* d, real* force, int size);
    void UnilateralFrictionResolution(int line, const MatXR & w, real* d, real* force, int size, real mu);
};

}