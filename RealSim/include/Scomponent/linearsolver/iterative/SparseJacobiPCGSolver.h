#pragma once

#include <fstream>
#include "Scomponent/linearsolver/iterative/SparseCGSolver.h"

namespace RealSim::linearsolver::iterative
{

class SparseJacobiPCGSolver: public SparseCGSolver {

public:
    SparseJacobiPCGSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = true)
            : SparseCGSolver(maxIter, tol, warmStart)
    {
        std::cout<<GREEN<<"Dense Linear Solver: SparseJacobiPCGSolver"<<RESET<<std::endl;
    }

protected:
    VecXR _jacobiPrecond;

    void computePrecond(const SpMatR &matrix) override
    {
        VecXR diagonal = matrix.diagonal();

        _jacobiPrecond.resize(diagonal.size());
        for(unsigned int i=0; i<diagonal.size(); i++)
        {
            if(diagonal[i] == 0.0)
            {
                std::cout<<RED<<"Error in SparseJacobiPCGSolver::computePrecond, diagonal element = 0.0, deactivate preconditioner"<<std::endl;
                _precond = false;
                return;
            }
            _jacobiPrecond[i] = 1.0 / diagonal[i];
        }
        _precond = true;
    }

    void applyPrecond(VecXR &x, const VecXR &y) override
    {
        RealSim::tools::linearalgebra::parallelDiagMV(x, _jacobiPrecond, y);
    }
};

}