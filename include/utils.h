#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Common helper macros for kernel execution configuration
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BLOCK_DIM 16

// Inline function for CUDA/CUBLAS error validation logic (CUDA sample style)
inline void cuda_check(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
                     cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

inline void cublas_check(cublasStatus_t result, const char* file, int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d: status code %d\n", file, line,
                     static_cast<int>(result));
        std::exit(EXIT_FAILURE);
    }
}

// Macros only for run cuda function
#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)
#define CUBLAS_CHECK(call) cublas_check((call), __FILE__, __LINE__)

// Helper utilities for memory allocation, initialization, and verification
void randomize_matrix(float* mat, int size);
void zero_matrix(float* mat, int size);
void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);
bool verify_matrix(const float* refC, const float* testC, int M, int N, float tolerance = 1e-2f);
float calc_max_abs_error(const float* refC, const float* testC, int M, int N);

#endif // UTILS_H
