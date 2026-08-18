#pragma once
#include "Scomponent/linearsolver/BaseSparseLinearSolver.h"
#include "Scomponent/tools/math/SparseMatrix.h"
#include "Scomponent/tools/math/SparseLDLT.h"
#include "tbb/tbb.h"
#include "Scomponent/tools/math/MatrixOperations.h"

namespace RealSim::linearsolver::direct
{

class SparseSingleJacobiSolver: public BaseSparseLinearSolver {

public:
    SparseSingleJacobiSolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: SparseSingleJacobiSolver"<<RESET<<std::endl;
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
        _R = (_Dinv * (Dmat - A)).sparseView();

        _csrR = CSMatrix(_R.transpose());
    }

    void solve_vec(VecXR &x, const VecXR &b) override
    {
        unsigned maxIter = 1;
        for(unsigned nb_iter=0; nb_iter<maxIter; nb_iter++)
        {
            //x = Dinv * b + R * x
            x = _Dinv * b + _R * x;
        }
    }

    void resetTimer() override { solve_timer.clear();}

    void printTimer() override {
        std::cout<<GREEN<<"SparseSingleJacobiSolver::solve, called by "<<BLUE<<solve_timer.times()<<GREEN<< " times, mean time cost = "<<BLUE<<solve_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
    }

protected:
    SpMatR _Dinv;
    SpMatR _R;
    RealSim::tools::linearalgebra::CSMatrix _csrR;

};

}