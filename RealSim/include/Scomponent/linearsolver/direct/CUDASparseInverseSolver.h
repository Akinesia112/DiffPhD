#pragma once
#include "Scomponent/linearsolver/direct/SparseInverseSolver.h"
#include "Scomponent/linearsolver/BaseCUDALinearSolver.h"
#include "Scomponent/tools/math/cuda_utils.h"

namespace RealSim::linearsolver::direct {
using namespace RealSim::tools::linearalgebra;

class CUDASparseInverseSolver : public SparseInverseSolver, public BaseCUDALinearSolver {

public:
    CUDASparseInverseSolver()
    {
        std::cout<<GREEN<<"Sparse Linear Solver: CUDASparseInverseSolver"<<RESET<<std::endl;
    }

    void init(SpMatR *sparseMatrix) override;

    int send_to_GPU();

    void solve_vec(VecXR &x, const VecXR &b) override;

    void addHAinvHT(MatXR &W, const CSMatrix & J) override;

    int solve_vec_gpu(CudaVec &x, CudaVec &b) override;

    int addHAinvHT_gpu(CudaMat &W, const CSMatrix & J) override;

protected:
    CudaCSMatrix cuda_ST; // ST in csr format (equivalent to S in csc format)
    CudaCSMatrix cuda_S; // S in csr format (equivalent to ST in csc format)
    CudaVec cuda_b, cuda_y, cuda_x;

    // for delasus
    CudaMat cuda_W;

    // for isodof
    int * d_ibar;

    CudaMat cuda_Wi;
    CudaMat cuda_JhWi;

    // for reuse isodof
    int * d_ibar_new;
    int * d_map;

    CudaCSMatrix cuda_JhT;

    CudaMat cuda_Wi_prev;
    CudaMat cuda_Wi_new;
    CudaMat cuda_ST_new;

    // for cuda library use
    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandleSpMV1 = NULL;
    cusparseHandle_t cusparseHandleSpMV2 = NULL;
    void *bufferSpMV1 = NULL;
    void *bufferSpMV2 = NULL;

    cusparseHandle_t     cusparseHandleSpMM = NULL;
    void*                bufferSpMM    = NULL;

    int free_delasus_gpu();
    int allocate_delasus_gpu(int size);
};

}
