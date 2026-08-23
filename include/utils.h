#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// =============================================================================
// 1. GPU Architecture & Vectorization Constants
// =============================================================================
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t SMEM_BANKS = 32;
constexpr uint32_t BANK_WIDTH_BYTES = 4;

// 128-bit Vectorization (float4)
constexpr uint32_t VEC_SIZE = 4;
constexpr uint32_t VEC_BYTES = sizeof(float) * VEC_SIZE; // 16 bytes

// Default Block Dimensions for basic kernels (Step 0 - Step 2)
constexpr uint32_t DEFAULT_BLOCK_DIM = 16;
constexpr uint32_t BLOCK_DIM = 16;

// =============================================================================
// 2. Kernel Launch & Math Helpers
// =============================================================================
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <typename T>
__host__ __device__ constexpr T ceil_div(T m, T n) {
    return (m + n - 1) / n;
}

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

// Vector math helpers for float4
__device__ inline float4 operator*(float a, float4 b) {
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__device__ inline float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

#endif // UTILS_H
