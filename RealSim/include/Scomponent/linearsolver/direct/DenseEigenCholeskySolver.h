#pragma once
#include "Scomponent/linearsolver/BaseDenseLinearSolver.h"

namespace RealSim::linearsolver::direct
{

class DenseEigenCholeskySolver: public BaseDenseLinearSolver {

public:
    DenseEigenCholeskySolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: DenseEigenCholeskySolver"<<RESET<<std::endl;
    }

    void init(MatXR * matrix) override
    {
        _matrix  = matrix;
        _denseCholesky.reset();
        _denseCholesky = std::make_shared<Eigen::LLT<MatXR>>(*matrix);
    }

    void solve_vec(VecXR &x, const VecXR &b) override
    {
        x = _denseCholesky->solve(b);
    }

    int getIterations() override {return 0;}

protected:
    std::shared_ptr<Eigen::LLT<MatXR>> _denseCholesky;

};

}