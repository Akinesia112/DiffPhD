#pragma once
#include "Scomponent/linearsolver/direct/SparseSingleJacobiSolver.h"
#include "Scomponent/linearsolver/BaseCUDALinearSolver.h"
#include "Scomponent/tools/math/cuda_utils.h"

namespace RealSim::linearsolver::direct
{

class CUDASparseSingleJacobiSolver: public SparseSingleJacobiSolver, public BaseCUDALinearSolver  {

public:
    CUDASparseSingleJacobiSolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: CUDASparseSingleJacobiSolver"<<RESET<<std::endl;
    }

    void init(SpMatR * sparseMatrix) override;

    int send_to_GPU();

    void solve_vec(VecXR &x, const VecXR &b) override;

    int solve_vec_gpu(CudaVec &x, CudaVec &b) override;

protected:
    CudaCSMatrix cuda_R;
    CudaVec cuda_b, cuda_x, cuda_y, cuda_Dinv;

    // for cuda library use
    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandleSpMV = NULL;
    void *bufferSpMV = NULL;
};

}