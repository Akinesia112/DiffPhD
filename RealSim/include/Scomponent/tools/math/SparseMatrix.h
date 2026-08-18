#pragma once
#include "types.h"

namespace RealSim::tools::linearalgebra
{

// row major format
class CSMatrix {
public :
    CSMatrix()
    {
        clearAll(0, 0);
    }

    CSMatrix(int n, int m, int nnz, const int * rowPtr, const int * colInd, const real * values)
    {
        clearAll(n, nnz);
        setSecondDim(m);

        memcpy(_outerPtr.data(), rowPtr, (_n+1) * sizeof(int));
        memcpy(_innerInd.data(), colInd, _nnz * sizeof(int));
        memcpy(_values.data(), values, _nnz * sizeof(real));
    }

    CSMatrix(const SpMatR& sparseMatrix)
    {
        createFromEigen(sparseMatrix);
    }

    CSMatrix switchOrder();

    void clearAll(int n, int nnz);
    void clearPattern(int n, int nnz);
    void clearValues(int n, int nnz);

    void setSecondDim(int m) { _m = m; }

    void print();

    SpMatR eigenFormat();

    void createFromEigen(const SpMatR& sparseMatrix);

    int _n, _m, _nnz;
    std::vector<int> _outerPtr;
    std::vector<int> _innerInd;
    std::vector<real> _values;
};





void switchCSMatrixOrder(CSMatrix & out, const CSMatrix & in);

inline void switchCSMatrixOrder(int n, int m, int nnz,
                                const int * outerPtr, const int * innerInd, const real * values,
                                int * tran_outerPtr, int * tran_innerInd, real * tran_values,
                                bool keepStructure)
{
    std::vector<int> tran_countvec;

    if (!keepStructure)
    {
        tran_countvec.clear();
        tran_countvec.resize(m);

        //First we count the number of value on each row.
        for (int j=0;j<nnz;j++) tran_countvec[innerInd[j]]++;

        //Now we make a scan to build tran_rowPtr
        tran_outerPtr[0] = 0;
        for (int j=0;j<m;j++) tran_outerPtr[j+1] = tran_outerPtr[j] + tran_countvec[j];
    }

    tran_countvec.clear();
    tran_countvec.resize(m);

    //we clear tran_countvec because we use it now to store how many values are written on each line
    for (int j = 0; j < n; j++)
    {
        for (int i = outerPtr[j]; i < outerPtr[j + 1]; i++)
        {
            const int line = innerInd[i];
            tran_innerInd[tran_outerPtr[line] + tran_countvec[line]] = j;
            tran_values[tran_outerPtr[line] + tran_countvec[line]] = values[i];
            tran_countvec[line]++;
        }
    }

}

}