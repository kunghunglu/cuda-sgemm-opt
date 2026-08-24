#ifndef SGEMM_H

#define SGEMM_H

#include <cublas_v2.h>

// Forward declarations for 10-step SGEMM optimization CUDA kernels & wrappers.
// Matrix conventions: A is M x K, B is K x N, C is M x N.
// Matrices are stored in Row-Major layout.
// C = alpha * (A * B) + beta * C  (Here default alpha = 1.0, beta = 0.0)

// Step 0: Naive Baseline
void run_sgemm_00_naive(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 1: Global Memory Coalescing
void run_sgemm_01_coalesced(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 2: Shared Memory Block Tiling (e.g. 32x32 tiles)
void run_sgemm_02_shared_mem(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 3: 1D Thread Tiling (Work per thread along 1 dimension)
void run_sgemm_03_1d_block_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 4: 2D Thread Tiling (Work per thread TM x TN outer products)
void run_sgemm_04_2d_block_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 5: Vectorized Memory Access (float4 loads/stores)
void run_sgemm_05_vectorized(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 6: Shared Memory Double Buffering / Software Pipelining
void run_sgemm_06_smem_double_buffering(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 7: Shared Memory Bank Conflict Free Layout (Transpose A + Row Padding)
void run_sgemm_07_bank_conflict_free(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 8: Hierarchical Warp Tiling (Block -> Warp -> Thread)
void run_sgemm_08_warp_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Step 9: Templated Hierarchical Warp Tiling & Architecture Tuning
void run_sgemm_09_templated_warp_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// Benchmark Reference: NVIDIA cuBLAS
void run_sgemm_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

#endif // SGEMM_H
