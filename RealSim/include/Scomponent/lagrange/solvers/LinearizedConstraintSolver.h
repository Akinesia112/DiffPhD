#pragma once
#include "Scomponent/lagrange/solvers/BaseConstraintSolver.h"

namespace RealSim::lagrange::constraintsolver
{

class LinearizedConstraintSolver : public BaseConstraintSolver
{
public:
    void init() override
    {
        unsigned int n = getVertexDimension();
        _xfree = MatX3R::Zero(n, 3);
    }

    void build(const MatX3R &pos, const MatX3R &b) override
    {
        if(_num_constraint == 0) return;

        build_timer.start();

        _systemlinearsolver->solve(_xfree, b);
        VecXR x_reshaped = _xfree.transpose().reshaped(_xfree.rows()*3, 1);
        _penefree = _jacobian * x_reshaped - _pene0;

        _penetration.setZero();
        _lambda.setZero();

        build_timer.stop();
    }

    void applyConstraintCorrection(MatX3R &x, MatX3R &b) override
    {
        if(_num_constraint == 0) return;

        correction_timer.start();

        VecXR db_reshaped = _jacobian.transpose() * _lambda;
        b += db_reshaped.reshaped(3, b.rows()).transpose();

        _systemlinearsolver->solve(x, b);

        correction_timer.stop();
    }

protected:
    MatX3R _xfree;
    VecXR _penefree;

};

}