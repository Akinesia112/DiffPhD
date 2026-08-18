#pragma once
#include "Scomponent/tools/math/SparseMatrix.h"
#include "tbb/tbb.h"

namespace RealSim::tools::linearalgebra
{

// implementation of LDLT decomposition in csparse

class SparseLDLTData {
public :
    void factorize(const CSMatrix & M);
    void solve(real * x, const real * b) const;
    void solveLower(real * x, const real * b) const;
    void computeLowerInverse();

    void resetMatrixValues(const CSMatrix & M);
    void resetMatrixPattern(const CSMatrix & M);
    void resetLPattern(int Lnnz);
    void resetSPattern(int Snnz);

    CSMatrix getA_CSFormat();
    CSMatrix getL_CSFormat();
    CSMatrix getS_CSFormat();
    SpMatR getA_EigenFormat();
    SpMatR getL_EigenFormat();
    SpMatR getS_EigenFormat();

    int n, M_nnz, L_nnz, S_nnz;

    //CS format for matrix M (symmetric), a copy of the matrix to invert
    std::vector<int> M_outerPtr, M_innerInd;
    std::vector<real> M_values;

    //CSC format for matrix LT, the upper triangular matrix of the LDL decomposition
    std::vector<int> LT_outerPtr, LT_innerInd;
    std::vector<real> LT_values;

    //CSC format for matrix L, the lower triangular matrix of the LDL decomposition
    std::vector<int> L_outerPtr, L_innerInd;
    std::vector<real> L_values;

    //CSC format for matrix S, the inverse of the lower triangular matrix L
    std::vector<int> S_outerPtr, S_innerInd;
    std::vector<real> S_values;

    //permutation
    std::vector<int> perm, invperm;

    //diagonal of D^-1
    std::vector<real> invD;

    std::vector<int> parent;

    bool keepstructure;
};

struct LowerInverseInRange {
    const int * S_outerPtr;
    const int * S_innerInd;
    real * S_values;
    const int * perm;
    const real * aL_values;

public:
    LowerInverseInRange(const int * _S_outerPtr, const int * _S_innerInd, real * _S_values, const int * _perm, const real * _aL_values)
            :S_outerPtr(_S_outerPtr), S_innerInd(_S_innerInd), S_values(_S_values), perm(_perm), aL_values(_aL_values) {}

    void operator()(const tbb::blocked_range<size_t>& r) const;
};

void LDLT_factorize(const CSMatrix & M, SparseLDLTData & res);

void LDLT_solve(real * x, const real * b, const SparseLDLTData & LDLT);

void LDLT_solveLower(real * x, const real * b, const SparseLDLTData & LDLT);

void LDLT_computeLowerInverse(SparseLDLTData & LDLT);

inline void LDLT_solve(unsigned n, real * x, const real * b, real * tmp,
                       const real * invD, const int * perm,
                       const int * LT_rowPtr, const int * LT_colInd, const real * LT_values,
                       const int * L_rowPtr, const int * L_colInd, const real * L_values);

inline void LDLT_solveLower(unsigned n, real * x, const real * b, const int * perm,
                            const int * LT_outerPtr, const int * LT_innerInd, const real * LT_values);

inline int LDLT_computeLowerInverseNNZ(unsigned n, const int * perm, const int * parent);

inline void LDLT_computeLowerInversePattern(unsigned n, int * S_outerPtr, int * S_innerInd, const int * perm, const int * parent);

inline void LDLT_computeAlignedLower(unsigned n, real * res, const int * S_outerPtr, const int * S_innerPtr, const int * perm,
                                     const int * L_outerPtr, const int * L_innerInd, const real * L_values);

void LDLT_computeLowerInverse(unsigned n, const int * S_outerPtr, const int * S_innerInd, real * S_values, const int * perm, const real * aL_values);

inline void LDLT_computeLowerInverse_line(int index, const int * S_outerPtr, const int * S_innerInd, real * S_values, const int * perm, const real * aL_values);

inline void LDLT_ordering(int n, int nnz, const int* M_outerPtr, const int* M_innerInd, int* perm, int* invperm);

inline void CSPARSE_symbolic (int n,const int * M_outerPtr, const int * M_innerInd, int * rowPtr,
                              const int * perm, const int * invperm, int * parent, int * Flag, int * Lnz);

inline void CSPARSE_numeric(int n, const int * M_outerPtr, const int * M_innerInd, const real * M_values,
                            const int * rowPtr, int * colInd, real * values, real * D,
                            const int * perm, const int * invperm, const int * parent,
                            int * Flag, int * Lnz, int * Pattern, real * Y);

inline bool compareMatrixPattern(int s_M, const int * M_colptr, const int * M_rowind,
                                 int s_P, const int * P_colptr, const int * P_rowind);

}