#pragma once
#include "Scomponent/linearsolver/BaseSparseLinearSolver.h"
#include "Scomponent/tools/math/SparseMatrix.h"
#include "Scomponent/tools/math/SparseLDLT.h"

namespace RealSim::linearsolver::direct
{

using namespace RealSim::tools::linearalgebra;

class SparseLDLTSolver: public BaseSparseLinearSolver {

public:
    SparseLDLTSolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: SparseLDLTSolver"<<RESET<<std::endl;
    }

    void init(SpMatR * sparseMatrix) override
    {
        _matrix  = sparseMatrix;

        std::cout<<"Prefactorizing the system matrix"<<std::endl;
        CSMatrix csrM(*sparseMatrix);
        _internalData.factorize(csrM);
    }

    void solve_vec(VecXR &x, const VecXR &b) override
    {
        _internalData.solve(x.data(), b.data());
    }

protected:
    SparseLDLTData _internalData;

};

}