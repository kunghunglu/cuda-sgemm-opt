#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err));                  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Macro for checking cuBLAS errors
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error at %s:%d: status code %d\n",   \
                         __FILE__, __LINE__, static_cast<int>(status));       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Helper utilities for memory allocation, initialization, and verification
void randomize_matrix(float* mat, int size);
void zero_matrix(float* mat, int size);
void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);
bool verify_matrix(const float* refC, const float* testC, int M, int N, float tolerance = 1e-2f);
float calc_max_abs_error(const float* refC, const float* testC, int M, int N);

#endif // UTILS_H
