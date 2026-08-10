#include "sgemm.h"
#include "utils.h"

/*
 * Step 1: Global Memory Coalescing
 * --------------------------------
 * Concept:
 * - Fixes the non-coalesced access pattern from Step 0 by swapping the thread-to-dimension mapping.
 * - Now threadIdx.x maps to contiguous column indices (N dimension) instead of rows.
 * - When 32 threads in a warp execute `B[k * N + col]`, thread 0 reads B[k, col_0],
 *   thread 1 reads B[k, col_0 + 1], ..., thread 31 reads B[k, col_0 + 31].
 * - This alignment allows the GPU memory controller to combine 32 x 4-byte loads into a single 128-byte memory transaction.
 * - Same fix applies to C writes: consecutive threads write to contiguous C addresses.
 * - For A, threads in the same row share the same row index, so `A[row * K + k]` is broadcast to all threads in that row.
 *
 * Performance impact:
 * - Dramatic improvement over Step 0 purely from memory access pattern change.
 * - No extra hardware resources (shared memory, registers) needed - just better addressing.
 */

__global__ void sgemm_01_coalesced_kernel(int M, int N, int K, float alpha,
                                          const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float beta,
                                          float* __restrict__ C) {
    // threadIdx.x maps to contiguous columns (N)
    // threadIdx.y maps to rows (M)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f; 
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void run_sgemm_01_coalesced(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));

    sgemm_01_coalesced_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
