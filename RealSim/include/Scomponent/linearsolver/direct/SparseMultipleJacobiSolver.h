#pragma once
#include "Scomponent/linearsolver/BaseSparseLinearSolver.h"
#include "Scomponent/tools/math/SparseMatrix.h"
#include "Scomponent/tools/math/SparseLDLT.h"
#include "tbb/tbb.h"
#include "Scomponent/tools/math/MatrixOperations.h"

namespace RealSim::linearsolver::direct
{

class SparseMultipleJacobiSolver: public BaseSparseLinearSolver {

public:
    SparseMultipleJacobiSolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: SparseMultipleJacobiSolver"<<RESET<<std::endl;
    }

    void init(SpMatR * sparseMatrix) override
    {
        _matrix  = sparseMatrix;

        MatXR A = _matrix->toDense();

        VecXR D = A.diagonal();

        _Dinv.resize(A.rows(), A.cols());
        for(int i=0; i<D.size(); i++)
        {
            _Dinv.coeffRef(i,i) = 1.0 / D[i];
        }


        MatXR Dmat = D.asDiagonal();
        MatXR R = _Dinv * (Dmat - A);

        std::cout<<"computing accumulating items ..."<<std::endl;

        unsigned l = 2;
        MatXR powerR = R;
        MatXR acc;
        acc.resize(A.rows(), A.cols());
        acc.setZero();
        acc.setIdentity();
        for(unsigned k=0; k<l-1; k++)
        {
            acc = acc + powerR;
            powerR = powerR * R;
        }

        std::cout<<"computing T and R ..."<<std::endl;

        acc = acc * _Dinv;
        _T = acc.sparseView();
        _R = powerR.sparseView();

        _csrT = CSMatrix(_T.transpose());
        _csrR = CSMatrix(_R.transpose());
    }

    void solve_vec(VecXR &x, const VecXR &b) override
    {
        unsigned maxIter = 1;
        for(unsigned nb_iter=0; nb_iter<maxIter; nb_iter++)
        {
            //x = T * b + R * x
            x = _T * b + _R * x;
        }
    }

    void resetTimer() override { solve_timer.clear();}

    void printTimer() override {
        std::cout<<GREEN<<"SparseMultipleJacobiSolver::solve, called by "<<BLUE<<solve_timer.times()<<GREEN<< " times, mean time cost = "<<BLUE<<solve_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
    }

protected:
    SpMatR _Dinv;
    SpMatR _R;
    SpMatR _T;
    RealSim::tools::linearalgebra::CSMatrix _csrR;
    RealSim::tools::linearalgebra::CSMatrix _csrT;

};

}