#pragma once
#include "Scomponent/linearsolver/BaseSparseLinearSolver.h"

namespace RealSim::linearsolver::direct
{

class SparseEigenCholeskySolver: public BaseSparseLinearSolver {

public:
    SparseEigenCholeskySolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: SparseEigenCholeskySolver"<<RESET<<std::endl;
    }

    void init(SpMatR * sparseMatrix) override
    {
        _matrix  = sparseMatrix;

//        std::cout<<"Prefactorizing the system matrix"<<std::endl;
        _sparseCholesky.reset();
        _sparseCholesky = std::make_shared<Eigen::SimplicialCholesky<SpMatR>>(*sparseMatrix);
    }

    void solve_vec(VecXR &x, const VecXR &b) override
    {
        x = _sparseCholesky->solve(b);
    }

protected:
    std::shared_ptr<Eigen::SimplicialCholesky<SpMatR>> _sparseCholesky;

};

}