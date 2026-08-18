#pragma once

#include <cmath>
#include "cuda.h"

namespace RealSim::tools::cuda
{

void cudaDiagMatrixVectorMultiplication(int n, const double * x, const double * D, double * y);

// dim( select(out) ) = k * n;
void cudaSelectSparseToDense(int k, int n, const int * select, const int * outerPtr, const int * innerInd, const double * values, double * out);

// out = select(A) * B^T
// dim( select(A) ) = k * n;
// dim(B) = l * n;
void cudaSelectSpMM(int k, int n, int l, const int * select, const int * A_outerPtr, const int * A_innerInd, const double * A_values, const double * B, double * out);

void cudaAssignWi(int k, int prev_size, const int * map, const double * Wi_prev, const double * Wi_new, double * out);

void cudaNonSmoothPrecondition(int n_pairs, int n_constraints, double dt, const int* ctype, const int* cid, const int* csize, const double * W, double * p);

void cudaNonSmoothFunctionFB(int n_pairs, double dt, const double * mu,
                             const int* ctype, const int* cid, const int* csize,
                             const double * precond, const double * penetration, const double * pene0, const double * lambda,
                             double * omega, double * compliance, double * h);

void cudaNonSmoothDelasus(int n_constraints, const double * W, const double * omega, const double * compliance, double * out);

bool cudaJacobiPrecondition(int n, const double * A, double * out);

void cudaBoundConstraintForces(int n_pairs, double max, const double * mu, const int* ctype, const int* cid, const int* csize, double * lambda);

}