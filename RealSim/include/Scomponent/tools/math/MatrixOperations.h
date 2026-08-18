#pragma once
#include "Scomponent/tools/math/SparseMatrix.h"
#include "Scomponent/tools/math/SparseLDLT.h"

namespace RealSim::tools::linearalgebra {

struct PeqInRange {
    VecXR &x;
    real alpha;
    const VecXR &y;
    real beta;
    const VecXR &z;

public:
    PeqInRange(VecXR &_x,
               real _alpha,
               const VecXR &_y,
               real _beta,
               const VecXR &_z)
            : x(_x),
              alpha(_alpha),
              y(_y),
              beta(_beta),
              z(_z) {
    }

    void operator()(const tbb::blocked_range<size_t> &r) const;
};

struct solveLowerInRange {
    MatXR *X;
    const MatXR *B;
    SparseLDLTData *_internalData;

public:
    solveLowerInRange(MatXR *_X,
                      const MatXR *_B,
                      SparseLDLTData *internalData)
            : X(_X),
              B(_B),
              _internalData(internalData) {
    }

    void operator()(const tbb::blocked_range<size_t> &r) const;
};

struct SpMVInRange {
    real *x;
    const int *outerPtr;
    const int *innerInd;
    const real *values;
    const real *y;

public:
    SpMVInRange(real *_x,
                const int *_outerPtr,
                const int *_innerInd,
                const real *_values,
                const real *_y)
            : x(_x),
              outerPtr(_outerPtr),
              innerInd(_innerInd),
              values(_values),
              y(_y) {
    }

    void operator()(const tbb::blocked_range<size_t> &r) const;
};


struct DnMVInRange {
    VecXR &x;
    const MatXR &M;
    const VecXR &y;

public:
    DnMVInRange(VecXR &_x,
                const MatXR &_M,
                const VecXR &_y)
            : x(_x),
              M(_M),
              y(_y) {
    }

    void operator()(const tbb::blocked_range<size_t> &r) const;
};

struct DiagMVInRange {
    real *x;
    const real *D;
    const real *y;

public:
    DiagMVInRange(real *_x,
                  const real *_D,
                  const real *_y)
            : x(_x),
              D(_D),
              y(_y) {
    }

    void operator()(const tbb::blocked_range<size_t> &r) const;
};

struct ComputeWiInRange {
    int n;
    int k;
    MatXR &Wi;
    const int *Ibar;
    const int *S_outerPtr;
    const int *S_innerInd;
    const real *S_values;

public:
    ComputeWiInRange(int _n,
                     int _k,
                     MatXR &Wi_,
                     const int *_Ibar,
                     const int *_S_outerPtr,
                     const int *_S_innerInd,
                     const real *_S_values)
            : n(_n),
              k(_k),
              Wi(Wi_),
              Ibar(_Ibar),
              S_outerPtr(_S_outerPtr),
              S_innerInd(_S_innerInd),
              S_values(_S_values){
    }

    void operator()(const tbb::blocked_range<size_t> &r) const;
};

struct ComputeWiReuseInRange {
    int n;
    int k;
    MatXR &Wi;
    const int *Ibar;
    const int *S_outerPtr;
    const int *S_innerInd;
    const real *S_values;
    const MatXR &Wi_prev;
    const int *map;
    int prev_size;

public:
    ComputeWiReuseInRange(int _n,
                          int _k,
                          MatXR &Wi_,
                          const int *_Ibar,
                          const int *_S_outerPtr,
                          const int *_S_innerInd,
                          const real *_S_values,
                          const MatXR &_Wi_prev,
                          const int *_map,
                          int _prev_size)
            : n(_n),
              k(_k),
              Wi(Wi_),
              Ibar(_Ibar),
              S_outerPtr(_S_outerPtr),
              S_innerInd(_S_innerInd),
              S_values(_S_values),
              Wi_prev(_Wi_prev),
              map(_map),
              prev_size(_prev_size) {
    }

    void operator()(const tbb::blocked_range<size_t> &r) const;
};

struct ComputeNonsmoothComplianceInRange {
    int c;
    MatXR &res;
    const MatXR &W;
    const VecXR &omega;
    const VecXR &compliance;

public:
    ComputeNonsmoothComplianceInRange(int _c,
                                      MatXR &_res,
                                      const MatXR &_W,
                                      const VecXR &_omega,
                                      const VecXR &_compliance)
            : c(_c),
              res(_res),
              W(_W),
              omega(_omega),
              compliance(_compliance) {
    }

    void operator()(const tbb::blocked_range<size_t> &r) const;
};


//////////////// vector operations///////////////
//x = alpha * y + beta * z;
void peq(VecXR &x, real alpha, const VecXR &y, real beta, const VecXR &z);

void parallelPeq(VecXR &x, real alpha, const VecXR &y, real beta, const VecXR &z);

//////////////// matrix operations///////////////
//x = M * y
void SpMV(VecXR &x, const CSMatrix &M, const VecXR &y);

void SpMV(int n, real *_x, const int *outerPtr, const int *innerInd, const real *values, const real *_y);

void parallelSpMV(VecXR &x, const CSMatrix &M, const VecXR &y);

void parallelSpMV(int n, real *_x, const int *outerPtr, const int *innerInd, const real *values,
                  const real *_y);

//x = (diagonal) D * y
void DiagMV(VecXR &x, const VecXR &D, const VecXR &y);

void parallelDiagMV(VecXR &x, const VecXR &D, const VecXR &y);

//X = M * Y
void SpMM(MatXR &X, const CSMatrix &M, const MatXR &Y);

void parallelSpMM(MatXR &X, const CSMatrix &M, const MatXR &Y);

void computeIsolatedDOFs(std::vector<int> &res_I, CSMatrix &res_Jh, const CSMatrix &J);

void computeWi(MatXR &Wi, const std::vector<int> &res_I, const CSMatrix &S);

void computeWi_line(int index, int n, int k, real *Wi_i, const int *Ibar, const int *S_outerPtr,
                    const int *S_innerInd, const real *S_values);

void computeWiNew(MatXR & WiNew, const std::vector<int> & I_bar, const std::vector<int> & I_bar_new, const CSMatrix & S);

void computeWiNew_line(int index, int n, int k_new, real *WiNew_i, const int *Ibar, const int * Ibar_new, const int *S_outerPtr,
                    const int *S_innerInd, const real *S_values);

void parallelComputeWi(MatXR &Wi, const std::vector<int> &res_I, const CSMatrix &S);

void parallelComputeWi(int _n, int _k, MatXR &Wi_, const int *_Ibar,
                       const int *_S_outerPtr, const int *_S_innerInd, const real *_S_values);

void parallelComputeWiReuse(MatXR &Wi, const std::vector<int> &res_I,
                            const CSMatrix &S, const MatXR &_Wi_prev,
                            const std::vector<int> & map, int prev_size);

void parallelComputeWiReuse(int n, int k, MatXR &Wi, const int *Ibar,
                            const int *S_outerPtr, const int *S_innerInd, const real *S_values,
                            const MatXR &Wi_prev,
                            const int * map, int prev_size);

//x = M * y
void DnMV(VecXR &x, const MatXR &M, const VecXR &y);

void parallelDnMV(VecXR &x, const MatXR &M, const VecXR &y);

void dropOutSmallValues(MatXR &matrix, real threshold, bool preserveDiag);

void computeReuseIsolatedDOFsMap(std::vector<int> &map, std::vector<int> &Ibar_new, const std::vector<int> &Ibar_prev, const std::vector<int> &Ibar);

}
