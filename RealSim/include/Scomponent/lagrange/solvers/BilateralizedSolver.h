#pragma once
#include "Scomponent/lagrange/solvers/LinearizedConstraintSolver.h"
#include "Scomponent/linearsolver/BaseDenseLinearSolver.h"

namespace RealSim::lagrange::constraintsolver
{

// Implementation of solver for bilaterized problem
// Linear Complementarity Problem (LCP) -> Linear Problem (LP)
//
// Original Author: Ziqiu ZENG
//
class BilateralizedSolver : public LinearizedConstraintSolver
{
public:
    typedef LinearizedConstraintSolver InheritSolver;

    void addConstraintLinearSolver(linearsolver::dense::TypeDenseLinearSolver type) override;

    void init() override;
    void solve() override;

protected:
    std::shared_ptr<linearsolver::BaseDenseLinearSolver> _constraintlinearsolver;

};

}