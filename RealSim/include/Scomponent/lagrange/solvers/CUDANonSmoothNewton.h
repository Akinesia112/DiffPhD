#pragma once
#include "Scomponent/lagrange/solvers/NonSmoothNewton.h"
#include "Scomponent/tools/math/cuda_utils.h"

namespace RealSim::lagrange::constraintsolver
{

using namespace RealSim::tools::cuda;

// Implementation of Non-smooth Newton method for contact (LCP) and frictional contact (NCP) constraint resolution
//
// Original Author: Ziqiu ZENG
//
class CUDANonSmoothNewtonSolver : public NonSmoothNewtonSolver
{
public:
    void init() override;

    void prepare(const std::vector<RealSim::contact::ContactPair> & contactPairSet) override;

    void build(const MatX3R &pos, const MatX3R &b) override;

    void solve() override;

    void applyConstraintCorrection(MatX3R &x, MatX3R &b) override;

    void post() override;

private:
    VecXR _b;

    CudaVec cuda_b;
    CudaVec cuda_b_copy;
    CudaVec cuda_x;
    CudaVec cuda_pos;

    CudaMat cuda_delasus;
    CudaMat cuda_nonsmooth_delasus;
    CudaVec cuda_precond;

    CudaVec cuda_pene0;
    CudaVec cuda_penetration;

    CudaVec cuda_omega;
    CudaVec cuda_compliance;
    CudaVec cuda_h;
    CudaVec cuda_rhs;

    CudaVec cuda_lambda;
    CudaVec cuda_dlambda;

    CudaVec cuda_tmp_c;

    CudaCSMatrix cuda_J;

    int * d_csize;
    int * d_ctype;
    int * d_cid;
    double * d_mu;

    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandleSpMVT = NULL;
    cusparseHandle_t cusparseHandleSpMVN = NULL;
    void *bufferSpMVT = NULL;
    void *bufferSpMVN = NULL;

    int init_gpu();

    int allocate_gpu(int num_constraint);

    int clear_gpu();

    int prepare_gpu(const std::vector<RealSim::contact::ContactPair> & contactPairSet);

    int build_gpu(CudaVec &pos, CudaVec &b);

    int applyConstraintCorrection_gpu(CudaVec &x, CudaVec &b);

    int computePreconditioner_gpu(CudaVec &precond, CudaMat &delasus);

    int computeNonsmoothFunction_gpu();

    int computeNonsmoothDelasus_gpu();

    int boundConstraintForces_gpu();

};

}