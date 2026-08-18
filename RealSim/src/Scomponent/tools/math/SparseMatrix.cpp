#include "Scomponent/tools/math/SparseMatrix.h"

namespace RealSim::tools::linearalgebra
{

void CSMatrix::clearAll(int n, int nnz)
{
    clearPattern(n, nnz);
    clearValues(n, nnz);
}

void CSMatrix::clearPattern(int n, int nnz)
{
    _n = n;
    _nnz = nnz;
    _outerPtr.clear(); _outerPtr.resize(n+1);
    _innerInd.clear(); _innerInd.resize(nnz);
}

void CSMatrix::clearValues(int n, int nnz)
{
    _n = n;
    _nnz = nnz;
    _values.clear(); _values.resize(nnz);
}

Eigen::SparseMatrix<real> CSMatrix::eigenFormat()
{
    Eigen::SparseMatrix<real> sparseMatrix(_m, _n);
    sparseMatrix.setZero();
    sparseMatrix.reserve(_nnz);

    memcpy(sparseMatrix.outerIndexPtr(), _outerPtr.data(), (_n+1) * sizeof(int));
    memcpy(sparseMatrix.innerIndexPtr(), _innerInd.data(), _nnz * sizeof(int));
    memcpy(sparseMatrix.valuePtr(), _values.data(), _nnz * sizeof(real));

    return sparseMatrix;
}

void CSMatrix::createFromEigen(const Eigen::SparseMatrix<real>& sparseMatrix)
{
    // Eigen store the sparse matrix in colum major by default, therefore we use the transpose
    clearAll(sparseMatrix.cols(), sparseMatrix.nonZeros());
    setSecondDim(sparseMatrix.rows());

    // Extract the inner data structures from the Eigen SparseMatrix
    const int * M_rowPtr = sparseMatrix.outerIndexPtr();
    const int * M_colInd = sparseMatrix.innerIndexPtr();
    const real * M_values = sparseMatrix.valuePtr();

    memcpy(_outerPtr.data(), M_rowPtr, (_n+1) * sizeof(int));
    memcpy(_innerInd.data(), M_colInd, _nnz * sizeof(int));
    memcpy(_values.data(), M_values, _nnz * sizeof(real));
}

CSMatrix CSMatrix::switchOrder()
{
    CSMatrix res;
    RealSim::tools::linearalgebra::switchCSMatrixOrder(res, *this);

    return res;
}

void switchCSMatrixOrder(CSMatrix & out, const CSMatrix & in)
{
    out.clearAll(in._m, in._nnz);
    out.setSecondDim(in._n);
    switchCSMatrixOrder(in._n, in._m, in._nnz, in._outerPtr.data(), in._innerInd.data(), in._values.data(),
                        out._outerPtr.data(), out._innerInd.data(), out._values.data(), false);
}

void CSMatrix::print()
{
    Eigen::SparseMatrix<real> sparseMatrix(_m, _n);
    sparseMatrix.setZero();
    sparseMatrix.reserve(_nnz);

    memcpy(sparseMatrix.outerIndexPtr(), _outerPtr.data(), (_n+1) * sizeof(int));
    memcpy(sparseMatrix.innerIndexPtr(), _innerInd.data(), _nnz * sizeof(int));
    memcpy(sparseMatrix.valuePtr(), _values.data(), _nnz * sizeof(real));

    std::cout<<sparseMatrix.toDense()<<std::endl;
}


}