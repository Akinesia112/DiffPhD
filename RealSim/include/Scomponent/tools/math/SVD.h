#pragma once
#include "types.h"

namespace RealSim::tools::svd
{

void signedEigenSVD(const Mat3R &F, Mat3R &U, Vec3R &sigma, Mat3R &V);

void signedMcAdamSVD(const Mat3R &F, Mat3R &U, Vec3R &sigma, Mat3R &V, Mat3R &S);

void eigen_svd_decompose(const Mat3R &F, Mat3R &U, Vec3R &sigma, Mat3R &V);

void McAdam_svd_decompose(const Mat3R &F, Mat3R &U, Vec3R &sigma, Mat3R &V, Mat3R &S);

void sofa_svd_decompose(const Mat3R &F, Mat3R &U, Vec3R &S, Mat3R &V);

void eigen_decomposition_iterative( const Mat3R &M, Mat3R &V, Vec3R &diag );

void QLAlgorithm( Vec3R &diag, Vec3R &subDiag, Mat3R &V );

real rabs(real r);

}