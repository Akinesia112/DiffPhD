#pragma once
#include <iostream>
#include "bench/BenchTimer.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "printconfig.h"
#include "Stimer.h"

#include "Scomponent/tools/math/SparseMatrix.h"

namespace RealSim::linearsolver
{

using namespace RealSim::tools::linearalgebra;

namespace sparse
{
enum TypeSparseLinearSolver {
    SPARSE_CHOLESKY_EIGEN,
    SPARSE_LDLT_CSPARSE,
    SPARSE_INVERSE,
    SPARSE_INVERSE_CUDA,
    SPARSE_SINGLE_JACOBI,
    SPARSE_MULTIPLE_JACOBI,
    SPARSE_SINGLE_JACOBI_CUDA,
    SPARSE_MULTIPLE_JACOBI_CUDA,
    SPARSE_CG,
    SPARSE_PCG_JACOBI,
    SPARSE_CR,
    SPARSE_PCR_JACOBI,
    SPARSE_PGS_COLORING
};

std::string type2String(TypeSparseLinearSolver m);
TypeSparseLinearSolver string2Type(std::string s);
}

class BaseSparseLinearSolver
{

public:
    virtual void init(SpMatR * matrix)
    {
        std::cout<<"BaseSparseLinearSolver::init(sparse) not implemented"<<std::endl;
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
        std::cout<<"BaseSparseLinearSolver::solve_vec() not implemented"<<std::endl;
    }

    void addHAinvHT(MatXR &W, const SpMatR & J)
    {
        addHAinvHT(W, CSMatrix(J));
    }

    virtual void addHAinvHT(MatXR &W, const CSMatrix & J)
    {
        std::cout<<"BaseSparseLinearSolver::addHAinvHT() not implemented"<<std::endl;
    }

    virtual void resetTimer() { solve_timer.clear(); }

    virtual void printTimer() {
        std::cout<<GREEN<<"BaseSparseLinearSolver::solve, called by "<<BLUE<<solve_timer.times()<<GREEN<< "times, mean time cost = "<<BLUE<<solve_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
    }

    const c_timer * getSolveTimer() {return &solve_timer;}
    const c_timer * getSchurTimer() {return &schur_timer;}

protected:
    SpMatR *_matrix;

    c_timer solve_timer;
    c_timer schur_timer;
};

}