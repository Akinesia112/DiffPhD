#pragma once

#include <fstream>
#include "Scomponent/linearsolver/iterative/DenseCGSolver.h"

namespace RealSim::linearsolver::iterative
{

class DenseJacobiPCGSolver: public DenseCGSolver {

public:
    DenseJacobiPCGSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = true)
            : DenseCGSolver(maxIter, tol, warmStart)
    {
        std::cout<<GREEN<<"Dense Linear Solver: DenseJacobiPCGSolver"<<RESET<<std::endl;
    }

protected:
    VecXR _jacobiPrecond;

    void computePrecond(const MatXR &matrix) override
    {
        VecXR diagonal = matrix.diagonal();

        _jacobiPrecond.resize(diagonal.size());
        for(unsigned int i=0; i<diagonal.size(); i++)
        {
            if(diagonal[i] == 0.0)
            {
                std::cout<<RED<<"Error in DenseJacobiPCGSolver::computePrecond, diagonal element = 0.0, deactivate preconditioner"<<RESET<<std::endl;
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