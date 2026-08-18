#pragma once

#include <fstream>
#include "Scomponent/linearsolver/iterative/CUDADenseCRSolver.h"
#include "Scomponent/tools/math/CUDAOperations.cuh"

namespace RealSim::linearsolver::iterative
{

class CUDADenseJacobiPCRSolver: public CUDADenseCRSolver {

public:
    CUDADenseJacobiPCRSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = true)
            : CUDADenseCRSolver(maxIter, tol, warmStart)
    {
        std::cout<<GREEN<<"Dense Linear Solver: CUDADenseJacobiPCRSolver"<<RESET<<std::endl;
    }

    int apply_precond_gpu(double * out, const double * in) override;

protected:
    VecXR _jacobiPrecond;
    double *d_precond = nullptr;

    void computePrecond(const MatXR &matrix) override; // cpu ver.

    // gpu api
    int init_gpu(int n) override;

    int set_gpu(CudaMat * matrix) override;

    int clear_gpu() override;

    int computePrecond_gpu();

    int send_precond_to_GPU();

};

}