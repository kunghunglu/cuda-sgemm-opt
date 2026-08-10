#include "sgemm.h"
#include "utils.h"

/*
 * Step 0: Naive SGEMM Kernel (Non-Coalesced)
 * -------------------------------------------
 * Concept:
 * - Each CUDA thread calculates one element C[row, col].
 * - Uses a 2D block and 2D grid structure.
 * - Iterates through K elements in global memory directly.
 *
 * Bottlenecks:
 * - Extremely low arithmetic intensity (FLOPs / byte loaded).
 * - Every element of A and B is re-read from global memory K times by neighboring threads.
 * - threadIdx.x maps to rows: consecutive threads in a warp access different rows of B and C,
 *   producing non-coalesced (strided) memory accesses. Each 4-byte load becomes a separate
 *   memory transaction instead of being merged into a single 128-byte transaction.
 * - Step 1 fixes this by swapping the mapping so threadIdx.x -> columns (coalesced).
 */

__global__ void sgemm_00_naive_kernel(int M, int N, int K, float alpha,
                                      const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float beta,
                                      float* __restrict__ C) {
    // NON-COALESCED: threadIdx.x maps to rows, threadIdx.y maps to columns.
    // Consecutive threads in a warp (varying threadIdx.x) access different rows of B and C,
    // causing strided memory accesses that cannot be merged into 128-byte transactions.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return ;
    }

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void run_sgemm_00_naive(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(CEIL_DIV(M, BLOCK_DIM), CEIL_DIV(N, BLOCK_DIM));

    sgemm_00_naive_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
