#pragma once
#include "Scomponent/linearsolver/direct/SparseMultipleJacobiSolver.h"
#include "Scomponent/linearsolver/BaseCUDALinearSolver.h"
#include "Scomponent/tools/math/cuda_utils.h"

namespace RealSim::linearsolver::direct
{

class CUDASparseMultipleJacobiSolver: public SparseMultipleJacobiSolver, public BaseCUDALinearSolver  {

public:
    CUDASparseMultipleJacobiSolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: CUDASparseMultipleJacobiSolver"<<RESET<<std::endl;
    }

    void init(SpMatR * sparseMatrix) override;

    int send_to_GPU();

    void solve_vec(VecXR &x, const VecXR &b) override;

    int solve_vec_gpu(CudaVec &x, CudaVec &b) override;

protected:
    CudaCSMatrix cuda_R;
    CudaCSMatrix cuda_T;
    CudaVec cuda_b, cuda_x, cuda_y;

    // for cuda library use
    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandleSpMVR = NULL;
    cusparseHandle_t cusparseHandleSpMVT = NULL;
    void *bufferSpMVR = NULL;
    void *bufferSpMVT = NULL;
};

}