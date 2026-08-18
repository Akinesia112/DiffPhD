#include "Scomponent/tools/math/SparseLDLT.h"
#include "metis.h"
#include <omp.h>
#include <thread>

namespace RealSim::tools::linearalgebra
{

// implementation of LDLT decomposition in csparse

void SparseLDLTData::factorize(const CSMatrix & M)
{
    LDLT_factorize(M, *this);
}

void SparseLDLTData::solve(real * x, const real * b) const
{
    LDLT_solve(x, b, *this);
}

void SparseLDLTData::solveLower(real * x, const real * b) const
{
    LDLT_solveLower(x, b, *this);
}

void SparseLDLTData::computeLowerInverse()
{
    LDLT_computeLowerInverse(*this);
}

void SparseLDLTData::resetMatrixValues(const CSMatrix &M)
{
    n = M._n;
    M_nnz = M._nnz;
    M_values.clear();
    M_values.resize(M_nnz);
    memcpy(M_values.data(), M._values.data(), M_nnz * sizeof(real));
}

void SparseLDLTData::resetMatrixPattern(const CSMatrix &M)
{
    perm.clear(); perm.resize(n);
    invperm.clear(); invperm.resize(n);
    parent.clear(); parent.resize(n);

    M_outerPtr.clear(); M_outerPtr.resize(n+1);
    M_innerInd.clear(); M_innerInd.resize(M_nnz);

    L_outerPtr.clear(); L_outerPtr.resize(n+1);
    LT_outerPtr.clear(); LT_outerPtr.resize(n+1);
    invD.clear(); invD.resize(n);

    S_outerPtr.clear(); S_outerPtr.resize(n+1);

    memcpy(M_outerPtr.data(), M._outerPtr.data(), (n+1) * sizeof(int));
    memcpy(M_innerInd.data(), M._innerInd.data(), M_nnz * sizeof(int));
}

void SparseLDLTData::resetLPattern(int Lnnz)
{
    L_nnz = Lnnz;
    L_innerInd.clear(); L_innerInd.resize(Lnnz);
    L_values.clear(); L_values.resize(Lnnz);
    LT_innerInd.clear(); LT_innerInd.resize(Lnnz);
    LT_values.clear(); LT_values.resize(Lnnz);
}

void SparseLDLTData::resetSPattern(int Snnz)
{
    S_nnz = Snnz;
    S_innerInd.clear(); S_innerInd.resize(Snnz);
    S_values.clear(); S_values.resize(Snnz);
}

CSMatrix SparseLDLTData::getA_CSFormat()
{
    auto A = CSMatrix(n, n, M_nnz, M_outerPtr.data(), M_innerInd.data(), M_values.data());
    return A;
}

CSMatrix SparseLDLTData::getL_CSFormat()
{
    auto L = CSMatrix(n, n, L_nnz, L_outerPtr.data(), L_innerInd.data(), L_values.data());
    return L;
}

CSMatrix SparseLDLTData::getS_CSFormat()
{
    auto S = CSMatrix(n, n, S_nnz, S_outerPtr.data(), S_innerInd.data(), S_values.data());
    return S;
}

Eigen::SparseMatrix<real> SparseLDLTData::getA_EigenFormat()
{
    auto A = getA_CSFormat();
    return A.eigenFormat();
}

Eigen::SparseMatrix<real> SparseLDLTData::getL_EigenFormat()
{
    auto L = getL_CSFormat();
    return L.eigenFormat();
}

Eigen::SparseMatrix<real> SparseLDLTData::getS_EigenFormat()
{
    auto S = getS_CSFormat();
    return S.eigenFormat();
}

void LDLT_factorize(const CSMatrix & M, SparseLDLTData & res)
{
    res.keepstructure = !(res.M_outerPtr.empty() ||res.M_innerInd.empty() ||
                          compareMatrixPattern(M._n, M._outerPtr.data(), M._innerInd.data(),
                                               res.n, (int*)res.M_outerPtr.data(), (int*)res.M_innerInd.data()));

    std::vector<real> Y;
    std::vector<int> Lnz,Flag,Pattern;

    res.resetMatrixValues(M);

    // we test if the matrix has the same struct as previous factorized matrix
    if (!res.keepstructure)
    {
        res.resetMatrixPattern(M);

        //ordering function
        LDLT_ordering(res.n, res.M_nnz, M._outerPtr.data(), M._innerInd.data(), res.perm.data(), res.invperm.data() );

        //symbolic factorization
        Lnz.clear(); Lnz.resize(res.n);
        Flag.clear(); Flag.resize(res.n);
        Pattern.clear(); Pattern.resize(res.n);

        CSPARSE_symbolic(res.n, res.M_outerPtr.data(), res.M_innerInd.data(), res.L_outerPtr.data(),
                         res.perm.data(),res.invperm.data(),res.parent.data(), Flag.data(),Lnz.data());

        res.resetLPattern(res.L_outerPtr[res.n]); // set nnz for L
    }

    const int * M_innerInd = res.M_innerInd.data();
    const int * M_outerPtr = res.M_outerPtr.data();
    const real * M_values = res.M_values.data();

    real * D = res.invD.data();

    int * L_outerPtr = res.L_outerPtr.data();
    int * L_innerInd = res.L_innerInd.data();
    real * L_values = res.L_values.data();

    //Numeric Factorization
    {
        Y.resize(res.n);
        CSPARSE_numeric(res.n, M_outerPtr, M_innerInd, M_values, L_outerPtr, L_innerInd, L_values, D,
                        res.perm.data(), res.invperm.data(), res.parent.data(),
                        Flag.data(),Lnz.data(),Pattern.data(),Y.data());

        //inverse the diagonal
        for (int i = 0; i < res.n; i++)
        {
            D[i] = 1 / D[i];
        }
    }

    // compute transpose
    switchCSMatrixOrder(res.n, res.n, res.L_nnz, L_outerPtr, L_innerInd, L_values,
                        res.LT_outerPtr.data(), res.LT_innerInd.data(), res.LT_values.data(), res.keepstructure);

}

void LDLT_solve(real * x, const real * b, const SparseLDLTData & LDLT)
{
    VecXR tmp = VecXR::Zero(LDLT.n);

    LDLT_solve(LDLT.n, x, b, tmp.data(), LDLT.invD.data(), LDLT.perm.data(),
               LDLT.L_outerPtr.data(), LDLT.L_innerInd.data(), LDLT.L_values.data(),
               LDLT.LT_outerPtr.data(), LDLT.LT_innerInd.data(), LDLT.LT_values.data());
}

void LDLT_solveLower(real * x, const real * b, const SparseLDLTData & LDLT)
{
    LDLT_solveLower(LDLT.n, x, b, LDLT.perm.data(),
                    LDLT.LT_outerPtr.data(), LDLT.LT_innerInd.data(), LDLT.LT_values.data());
}

inline void LDLT_solveLower(unsigned n, real * x, const real * b, const int * perm,
                            const int * LT_outerPtr, const int * LT_innerInd, const real * LT_values) {

    //lower solution
    for (int j = 0 ; j < n ; j++) {
        real acc = b[perm[j]];
        for (int p = LT_outerPtr [j] ; p < LT_outerPtr[j+1] ; p++) {
            acc -= LT_values[p] * x[LT_innerInd[p]];
        }
        x[j] = acc;
    }
}

inline void LDLT_solve(unsigned n, real * x, const real * b, real * tmp,
                       const real * invD, const int * perm,
                       const int * L_outerPtr, const int * L_innerInd, const real * L_values,
                       const int * LT_outerPtr, const int * LT_innerInd, const real * LT_values) {

    //lower solution
    for (int j = 0 ; j < n ; j++) {
        real acc = b[perm[j]];
        for (int p = LT_outerPtr [j] ; p < LT_outerPtr[j+1] ; p++) {
            acc -= LT_values[p] * tmp[LT_innerInd[p]];
        }
        tmp[j] = acc;
    }

    //upper solution
    for (int j = n-1 ; j >= 0 ; j--) {
        tmp[j] *= invD[j];

        for (int p = L_outerPtr[j] ; p < L_outerPtr[j+1] ; p++) {
            tmp[j] -= L_values[p] * tmp[L_innerInd[p]];
        }

        x[perm[j]] = tmp[j];
    }
}


void LDLT_computeLowerInverse(SparseLDLTData & LDLT)
{
    int S_nnz = LDLT_computeLowerInverseNNZ(LDLT.n, LDLT.invperm.data(), LDLT.parent.data());
    LDLT.resetSPattern(S_nnz);
    LDLT_computeLowerInversePattern(LDLT.n, LDLT.S_outerPtr.data(), LDLT.S_innerInd.data(), LDLT.invperm.data(), LDLT.parent.data());

    // compute aligned L
    std::vector<real> aL_values;
    aL_values.clear(); aL_values.resize(S_nnz, 0.0);

    LDLT_computeAlignedLower(LDLT.n, aL_values.data(), LDLT.S_outerPtr.data(), LDLT.S_innerInd.data(), LDLT.invperm.data(),
                             LDLT.L_outerPtr.data(), LDLT.L_innerInd.data(), LDLT.L_values.data());

    LDLT_computeLowerInverse(LDLT.n, LDLT.S_outerPtr.data(), LDLT.S_innerInd.data(), LDLT.S_values.data(), LDLT.perm.data(), aL_values.data());
}

inline int LDLT_computeLowerInverseNNZ(unsigned n, const int * invperm, const int * parent) {

    int count = 0;
    for (int i = 0 ; i < n ; i++) {
        // compute nnz for S
        int index = invperm[i];

        int innercount = 1;
        while(parent[index] != -1)
        {
            innercount++;
            index = parent[index];
        }

        count += innercount;
    }

    return count;
}

inline void LDLT_computeLowerInversePattern(unsigned n, int * S_outerPtr, int * S_innerInd, const int * invperm, const int * parent) {

    int count = 0;
    S_outerPtr[0] = count;

    for (int i = 0 ; i < n ; i++) {
        //compute CSC pattern for S
        int index = invperm[i];

        int innercount = 1;
        S_innerInd[count] = index;
        while(parent[index] != -1)
        {
            S_innerInd[count + innercount] = parent[index];
            innercount++;
            index = parent[index];
        }

        count += innercount;
        S_outerPtr[i+1] = count;
    }
}

//compute aligned L (to S), return the value ptr of aligned L
inline void LDLT_computeAlignedLower(unsigned n, real * res, const int * S_outerPtr, const int * S_innerPtr, const int * invperm,
                                     const int * L_outerPtr, const int * L_innerInd, const real * L_values){
    for (int row = 0 ; row < n ; row++)
    {
        int invr = invperm[row];

        int ptrS = S_outerPtr[row];
        for(int ptrL = L_outerPtr[invr]; ptrL < L_outerPtr[invr+1]; ptrL++)
        {
            while(ptrS < S_outerPtr[row+1])
            {
                if(S_innerPtr[ptrS] == L_innerInd[ptrL])
                {
                    res[ptrS] = L_values[ptrL];
                    break;
                }
                ptrS++;
            }
        }
    }
}

inline void LDLT_computeLowerInverse_line(int index, const int * S_outerPtr, const int * S_innerInd, real * S_values, const int * perm, const real * aL_values) {
    S_values[S_outerPtr[index]] = 1.0;

    for(int i = S_outerPtr[index]+1; i < S_outerPtr[index+1]; i++)
    {
        int col = perm[S_innerInd[i-1]];

        for(int j = i, k = S_outerPtr[col]+1; (j < S_outerPtr[index+1]) || (k < S_outerPtr[col+1]); j++, k++)
        {
            S_values[j] -= aL_values[k] * S_values[i-1];
        }
    }
}

void LowerInverseInRange::operator()(const tbb::blocked_range<size_t>& r) const {
    for( size_t index=r.begin(); index!=r.end(); ++index )
    {
        S_values[S_outerPtr[index]] = 1.0;

        for(int i = S_outerPtr[index]+1; i < S_outerPtr[index+1]; i++)
        {
            int col = perm[S_innerInd[i-1]];

            for(int j = i, k = S_outerPtr[col]+1; (j < S_outerPtr[index+1]) || (k < S_outerPtr[col+1]); j++, k++)
            {
                S_values[j] -= aL_values[k] * S_values[i-1];
            }
        }
    }
}

void LDLT_computeLowerInverse(unsigned n, const int * S_outerPtr, const int * S_innerInd, real * S_values, const int * perm, const real * aL_values) {
    //lower solution
//    for (int index = 0 ; index < n ; index++) {
//        LDLT_computeLowerInverse_line(index, S_outerPtr, S_innerInd, S_values, perm, aL_values);
//    }

    // parallel solution
    tbb::parallel_for(tbb::blocked_range<size_t>(0,n), LowerInverseInRange(S_outerPtr, S_innerInd, S_values, perm, aL_values));
}


inline void LDLT_ordering(int n, int nnz, const int* M_outerPtr, const int* M_innerInd, int* perm, int* invperm)
{
    //TODO: use metis nested dissection to compute permutations

    std::vector<int> tran_countvec, t_xadj, t_adj, xadj, adj;

    tran_countvec.clear();
    tran_countvec.resize(n);

    //First we count the number of value on each row.
    for (int j=0;j<n;j++) {
        for (int i=M_outerPtr[j];i<M_outerPtr[j+1];i++) {
            int col = M_innerInd[i];
            if (col>j) tran_countvec[col]++;
        }
    }

    //Now we make a scan to build tran_outerPtr
    t_xadj.resize(n+1);
    t_xadj[0] = 0;
    for (int j=0;j<n;j++) t_xadj[j+1] = t_xadj[j] + tran_countvec[j];

    //we clear tran_countvec becaus we use it now to stro hown many value are written on each line
    tran_countvec.clear();
    tran_countvec.resize(n);

    t_adj.resize(t_xadj[n]);
    for (int j=0;j<n;j++) {
        for (int i=M_outerPtr[j];i<M_outerPtr[j+1];i++) {
            int line = M_innerInd[i];
            if (line>j) {
                t_adj[t_xadj[line] + tran_countvec[line]] = j;
                tran_countvec[line]++;
            }
        }
    }

    adj.clear();
    xadj.resize(n+1);
    xadj[0] = 0;
    for (int j=0; j<n; j++)
    {
        //copy the lower part
        for (int ip = t_xadj[j]; ip < t_xadj[j+1]; ip++) {
            adj.push_back(t_adj[ip]);
        }

        //copy only the upper part
        for (int ip = M_outerPtr[j]; ip < M_outerPtr[j+1]; ip++) {
            int col = M_innerInd[ip];
            if (col > j) adj.push_back(col);
        }

        xadj[j+1] = adj.size();
    }

//    for(int i=0; i<n; i++)
//    {
//        perm[i] = i;
//        invperm[i] = i;
//    }

    METIS_NodeND(&n, xadj.data(), adj.data(), NULL, NULL, perm,invperm);

//    int blocksize=100;
//    int nbpart=n/blocksize;
//
//    int nb_level=0;
//    while(pow(2, nb_level) < nbpart){
//        nb_level++;
//    }
//
//    int nbpartitions=pow(2, nb_level-1);
//    const int size = 2*nbpartitions;
//
//    std::vector<int> psizes;
//    psizes.resize(size);
//    METIS_NodeNDP(n, xadj.data(), adj.data(), NULL, nbpartitions, NULL, perm,invperm, psizes.data());



}

inline void CSPARSE_symbolic (int n,const int * M_outerPtr, const int * M_innerInd, int * outerPtr,
                              const int * perm, const int * invperm, int * parent, int * Flag, int * Lnz)
{
    for (int k = 0 ; k < n ; k++)
    {
        parent [k] = -1 ;	    // parent of k is not yet known
        Flag [k] = k ;		    // mark node k as visited
        Lnz [k] = 0 ;		    // count of nonzeros in column k of L
        const int kk = perm[k];  // kth original, or permuted, column
        for (int p = M_outerPtr[kk] ; p < M_outerPtr[kk+1] ; p++)
        {
            // A (i,k) is nonzero (original or permuted A)
            int i = invperm[M_innerInd[p]];
            if (i < k)
            {
                // follow path from i to root of etree, stop at flagged node
                for ( ; Flag [i] != k ; i = parent [i])
                {
                    // find parent of i if not yet determined
                    if (parent [i] == -1) parent [i] = k ;
                    Lnz [i]++ ;				// L (k,i) is nonzero
                    Flag [i] = k ;			// mark i as visited
                }
            }
        }
    }

    outerPtr[0] = 0 ;
    for (int k = 0 ; k < n ; k++) outerPtr[k+1] = outerPtr[k] + Lnz[k] ;
}

inline void CSPARSE_numeric(int n, const int * M_outerPtr, const int * M_innerInd, const real * M_values,
                            const int * outerPtr, int * innerInd, real * values, real * D,
                            const int * perm, const int * invperm, const int * parent,
                            int * Flag, int * Lnz, int * Pattern, real * Y)
{
    real yi, l_ki ;
    int i, p, kk, len, top ;

    for (int k = 0 ; k < n ; k++)
    {
        Y [k] = 0.0 ;		    // Y(0:k) is now all zero
        top = n ;		    // stack for pattern is empty
        Flag [k] = k ;		    // mark node k as visited
        Lnz [k] = 0 ;		    // count of nonzeros in column k of L
        kk = perm[k];  // kth original, or permuted, column
        for (p = M_outerPtr[kk] ; p < M_outerPtr[kk+1] ; p++)
        {
            i = invperm[M_innerInd[p]];	// get A(i,k)
            if (i <= k)
            {
                Y[i] += M_values[p] ;  // scatter A(i,k) into Y (sum duplicates)
                for (len = 0 ; Flag[i] != k ; i = parent[i])
                {
                    Pattern [len++] = i ;   // L(k,i) is nonzero
                    Flag [i] = k ;	    // mark i as visited
                }
                while (len > 0) Pattern[--top] = Pattern [--len] ;
            }
        }
        // compute numerical values kth row of L (a sparse triangular solve)
        D[k] = Y [k] ;		    // get D(k,k) and clear Y(k)
        Y[k] = 0.0 ;
        for ( ; top < n ; top++)
        {
            i = Pattern [top] ;	    // Pattern [top:n-1] is pattern of L(:,k)
            yi = Y [i] ;	    // get and clear Y(i)
            Y [i] = 0.0 ;
            for (p = outerPtr[i] ; p < outerPtr[i] + Lnz [i] ; p++)
            {
                Y[innerInd[p]] -= values[p] * yi ;
            }
            l_ki = yi / D[i] ;	    // the nonzero entry L(k,i)
            D[k] -= l_ki * yi ;
            innerInd[p] = k ;	    // store L(k,i) in column form of L
            values[p] = l_ki ;
            Lnz[i]++ ;		    // increment count of nonzeros in col i
        }

        if (D[k] == 0.0)
        {
            std::cerr<< "SparseLDLT Failed to factorize, D(k,k) is zero"<<std::endl; ;
            return;
        }
    }
}

inline bool compareMatrixPattern(int s_M, const int * M_outerPtr, const int * M_innerInd,
                                 int s_P, const int * P_outerPtr, const int * P_innerInd)
{
    if (s_M != s_P) return true;
    if (M_outerPtr[s_M] != P_outerPtr[s_M] ) return true;

    for (int i=0;i<s_P;i++) {
        if (M_outerPtr[i]!=P_outerPtr[i]) return true;
    }

    for (int i=0;i<M_outerPtr[s_M];i++) {
        if (M_innerInd[i]!=P_innerInd[i]) return true;
    }

    return false;
}


}