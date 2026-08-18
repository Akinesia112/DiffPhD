#pragma once
#include "Scomponent/linearsolver/BaseSparseLinearSolver.h"
#include "Scomponent/tools/math/SparseMatrix.h"
#include "Scomponent/tools/math/SparseLDLT.h"
#include "tbb/tbb.h"
#include "Scomponent/tools/math/MatrixOperations.h"

namespace RealSim::linearsolver::direct
{

class SparseInverseSolver: public BaseSparseLinearSolver {

public:
    SparseInverseSolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: SparseInverseSolver"<<RESET<<std::endl;
    }

    void init(SpMatR * sparseMatrix) override;

    void solve_vec(VecXR &x, const VecXR &b) override;

    void solve_mat(MatXR &X, const MatXR &B);

    void addHAinvHT(MatXR &W, const CSMatrix & J) override;

    void resetTimer() override { solve_timer.clear(); schur_timer.clear();}

    void printTimer() override {
        std::cout<<GREEN<<"SparseInverseSolver::solve, called by "<<BLUE<<solve_timer.times()<<GREEN<< " times, mean time cost = "<<BLUE<<solve_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;

        if(schur_timer.times() != 0)
            std::cout<<GREEN<<"SparseInverseSolver::addHAinvHT, called by "<<BLUE<<schur_timer.times()<<GREEN<< " times, mean time cost = "<<BLUE<<schur_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
    }

protected:
    RealSim::tools::linearalgebra::SparseLDLTData _internalData;

    VecXR _Dinv;
    VecXR _sqrtDinv;

    RealSim::tools::linearalgebra::CSMatrix _csrS;
    RealSim::tools::linearalgebra::CSMatrix _csrST;

    VecXR _Sb;

    // for isodof
    std::vector<int> _Ibar;
    RealSim::tools::linearalgebra::CSMatrix _Jh;
    MatXR _Wi;
    MatXR _Wi_prev;
    MatXR _JhWi;

    // for reuse isodof
    std::vector<int> _map;
    std::vector<int> _Ibar_new;
    std::vector<int> _Ibar_prev;

    void computeLowerInverse_fast();
    void computeLowerInverse_slow();

    void solveLower(MatXR &X, const MatXR &B);
    void parallelSolveLower(MatXR &X, const MatXR &B);

    void DiagMSpM(const real * D, RealSim::tools::linearalgebra::CSMatrix & M);
    
    void analysisContact(const RealSim::tools::linearalgebra::CSMatrix & J);
};

}