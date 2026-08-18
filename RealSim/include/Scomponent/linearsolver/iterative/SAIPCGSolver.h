//#pragma once
//
//#include <fstream>
//#include "Scomponent/linearsolver/iterative/BaseIterativeSolver.h"
//
//namespace RealSim::linearsolver::iterative
//{
//
//class SAIPCGSolver: public CGSolver {
//
//public:
//    typedef CGSolver InheritSolver;
//
//    SAIPCGSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = true)
//        : CGSolver(maxIter, tol, warmStart){}
//
//protected:
//    MatXR _sparseInverse;
//
//    void computePrecond(const SpMatR &matrix) override
//    {
//        std::cout << "Initialize PCG: computing sparse LLT factorization" << std::endl;
//
//        Eigen::SimplicialCholesky<SpMatR> cholesky(matrix);
//        MatXR identity = MatXR::Identity(matrix.rows(), matrix.cols());
//        _sparseInverse = cholesky.solve(identity);
//
//        std::cout << "NNZ in _sparseInverse: " << (real) _sparseInverse.count() / (real) _sparseInverse.size()
//                  << std::endl;
//        real threshold = 1e-3;
//        RealSim::tools::linearalgebra::dropOutSmallValues(_sparseInverse, threshold, true);
//
//        std::cout << "NNZ in _sparseInverse after filtering: "
//                  << (real) _sparseInverse.count() / (real) _sparseInverse.size() << std::endl;
//
//        unsigned int nnz_L = cholesky.rawMatrix().nonZeros();
//        std::cout << "NNZ in L: " << (real) nnz_L / (real) matrix.size() << std::endl;
//        std::cout << "NNZ in A: " << (real) matrix.nonZeros() / (real) matrix.size() << std::endl;
//
//        _precond = true;
//    }
//
//};
//
//}