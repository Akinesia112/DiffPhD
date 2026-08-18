/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cmath>
#include <functional>
#include <iostream>

#include <cuda_runtime_api.h>

#include <cublas_v2.h>
#include <cusparse.h>
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <assert.h>

#include "types.h"
#include "Scomponent/tools/math/SparseMatrix.h"

// CUDA API error checking
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

namespace RealSim::tools::cuda
{

typedef struct CudaVecStruct {
    cusparseDnVecDescr_t    vec;
    double*                 ptr;

    int setZero(int n)
    {
        CHECK_CUDA( cudaMemset(ptr, 0.0, n * sizeof(double)));
        return EXIT_SUCCESS;
    }
} CudaVec;

typedef struct CudaMatStruct {
    cusparseDnMatDescr_t    mat;
    double*                 ptr;
} CudaMat;

typedef struct CudaCSMatrixStruct {
    cusparseSpMatDescr_t    mat;
    int*                    outerPtr;
    int*                    innerInd;
    double*                 values;
} CudaCSMatrix;


int send_Mat_to_CPU(RealSim::MatXR &out, CudaMat &in);

int send_Mat_to_GPU(CudaMat &out, RealSim::MatXR &in);

int send_CSMat_to_CPU(RealSim::tools::linearalgebra::CSMatrix &out, CudaCSMatrix &in);

int send_CSMat_to_GPU(CudaCSMatrix &out, RealSim::tools::linearalgebra::CSMatrix &in);

int send_Vec_to_CPU(RealSim::VecXR &out, CudaVec &in);

int send_Vec_to_GPU(CudaVec &out, const RealSim::VecXR &in);

}