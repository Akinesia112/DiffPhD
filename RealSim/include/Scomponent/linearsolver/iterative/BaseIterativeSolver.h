#pragma once
#include "Scomponent/tools/math/MatrixOperations.h"

namespace RealSim::linearsolver::iterative
{

class BaseIterativeSolver{

public:
    BaseIterativeSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = true)
        :_maxIter(maxIter), _tol(tol), _warmStart(warmStart){}

    void setParameters(unsigned int maxIter, real tol, bool warmStart)
    {
        _maxIter = maxIter;
        _tol = tol;
        _warmStart = warmStart;
    }

protected:
    int _nb_iter;

    unsigned int _maxIter;
    real _tol;
    bool _warmStart;

    bool _precond;
};

}