#pragma once
#include <iostream>
#include "bench/BenchTimer.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "printconfig.h"
#include "Stimer.h"

namespace RealSim::linearsolver
{

namespace dense {
    enum TypeDenseLinearSolver {
        DENSE_CHOLESKY_EIGEN,
        DENSE_CG,
        DENSE_PCG_JACOBI,
        DENSE_CR,
        DENSE_CR_CUDA,
        DENSE_PCR_JACOBI,
        DENSE_PCR_JACOBI_CUDA
    };

    std::string type2String(TypeDenseLinearSolver m);
    TypeDenseLinearSolver string2Type(std::string s);
}

class BaseDenseLinearSolver
{

public:
    virtual void init(MatXR * matrix)
    {
        std::cout<<"BaseDenseLinearSolver::init() not implemented"<<std::endl;
    }

    virtual void set(MatXR * matrix)
    {
        std::cout<<"BaseDenseLinearSolver::set() not implemented"<<std::endl;
    }

    void solve(MatX3R &x, const MatX3R &b)
    {
        Eigen::BenchTimer timer;
        VecXR b_reshaped = b.transpose().reshaped(b.rows()*3, 1);
        VecXR x_reshaped = x.transpose().reshaped(x.rows()*3, 1);

        solve_vec(x_reshaped, b_reshaped);

        x = x_reshaped.reshaped(3, x.rows()).transpose();
    }

    virtual void solve_vec(VecXR &x, const VecXR &b)
    {
        std::cout<<"BaseDenseLinearSolver::solve() not implemented"<<std::endl;
    }

    virtual void addHAinvHT(MatXR &W, const SpMatR & J)
    {
        std::cout<<"BaseDenseLinearSolver::addHAinvHT() not implemented"<<std::endl;
    }

    virtual void computeNonsmoothDelasus(MatXR & res, const MatXR & W, const VecXR & omega, const VecXR & compliance)
    {
        std::cout<<"BaseDenseLinearSolver::computeNonsmoothDelasus() not implemented"<<std::endl;
    }

    virtual void clear()
    {
        // do nothing
    }

    virtual int getIterations() = 0;

protected:
    MatXR * _matrix;
};

}