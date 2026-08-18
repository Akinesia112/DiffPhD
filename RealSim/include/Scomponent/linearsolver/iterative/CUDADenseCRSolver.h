#pragma once
#include "Scomponent/linearsolver/iterative/DenseCRSolver.h"
#include "Scomponent/linearsolver/BaseCUDALinearSolver.h"

#include "Scomponent/tools/math/cuda_utils.h"

namespace RealSim::linearsolver::iterative
{

class CUDADenseCRSolver: public DenseCRSolver, public BaseCUDALinearSolver  {
public:

    CUDADenseCRSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = false)
            : DenseCRSolver(maxIter, tol, warmStart)
    {
        std::cout<<GREEN<<"Dense Linear Solver: CUDADenseCRSolver"<<RESET<<std::endl;
    }

    int _n;

    void init(MatXR * matrix) override;

    void set(MatXR * matrix) override;

    void clear() override;
    
    int send_to_GPU();

    void solve_vec(VecXR &x, const VecXR &b) override;

    virtual int apply_precond_gpu(double * out, const double * in);

protected:
    cublasHandle_t   cublasHandle   = NULL;

    double *d_A = nullptr;
    CudaVec cuda_b;
    CudaVec cuda_x;

    double *d_r = nullptr;
    double *d_d = nullptr;
    double *d_q = nullptr;
    double *d_h = nullptr;
    double *d_s = nullptr;

    int init_gpu(int n) override;

    int set_gpu(CudaMat * matrix) override;

    int solve_vec_gpu(CudaVec &x, CudaVec &b) override;

    int clear_gpu() override;


    int allocate_gpu(int n);

    int allocate_gpu_auxliary(int n);

    int clear_gpu_auxilary();
};

}