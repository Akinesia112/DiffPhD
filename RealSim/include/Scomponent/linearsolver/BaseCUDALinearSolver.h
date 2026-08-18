#pragma once
#include <iostream>
#include "bench/BenchTimer.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "printconfig.h"
#include "Stimer.h"
#include "Scomponent/tools/math/cuda_utils.h"
#include "Scomponent/tools/math/SparseMatrix.h"

namespace RealSim::linearsolver
{

using namespace RealSim::tools::cuda;

class BaseCUDALinearSolver
{

public:
    void init(int n)
    {
        init_gpu(n);
    }

    void set(CudaMat * matrix)
    {
        set_gpu(matrix);
    }

    void solve(CudaVec &x, CudaVec &b)
    {
        solve_vec_gpu(x, b);
    }

    void addHAinvHT(CudaMat &W, const RealSim::tools::linearalgebra::CSMatrix & J)
    {
        addHAinvHT_gpu(W, J);
    }

    void free_gpu()
    {
        clear_gpu();
    }

protected:
    virtual int init_gpu(int n)
    {
        std::cout<<"BaseCUDALinearSolver::init_gpu() not implemented"<<std::endl;

        return EXIT_SUCCESS;
    }

    virtual int set_gpu(CudaMat * matrix)
    {
        std::cout<<"BaseCUDALinearSolver::set_gpu() not implemented"<<std::endl;

        return EXIT_SUCCESS;
    }

    virtual int solve_vec_gpu(CudaVec &x, CudaVec &b)
    {
        std::cout<<"BaseCUDALinearSolver::solve() not implemented"<<std::endl;

        return EXIT_SUCCESS;
    }

    virtual int addHAinvHT_gpu(CudaMat &W, const RealSim::tools::linearalgebra::CSMatrix & J)
    {
        std::cout<<"BaseCUDALinearSolver::addHAinvHT_cuda() not implemented"<<std::endl;

        return EXIT_SUCCESS;
    }

    virtual int clear_gpu()
    {
        std::cout<<"BaseCUDALinearSolver::clear_gpu() not implemented"<<std::endl;

        return EXIT_SUCCESS;
    }

};

}