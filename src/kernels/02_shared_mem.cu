#include "sgemm.h"
#include "utils.h"

/*
 * Step 2: Shared Memory Block Tiling
 * ----------------------------------
 * Concept:
 * - Divides matrices A, B, C into tiles of size BLOCK_DIM x BLOCK_DIM.
 * - A block of BLOCK_DIM x BLOCK_DIM threads loads one tile of A and one tile of B into Shared Memory (__shared__).
 * - Synchronizes threads using __syncthreads().
 * - Computes matrix multiplication on fast shared memory instead of slow global memory.
 * - Global memory access frequency drops from O(K) per element to O(K / BLOCK_DIM).
 */

__global__ void sgemm_02_shared_mem_kernel(int M, int N, int K, float alpha,
                                            const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float beta,
                                            float* __restrict__ C) {
    __shared__ float As[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM];

    uint32_t row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    uint32_t col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    float sum = 0.0f;

    // Loop over sub-tiles along K dimension
    uint32_t numTiles = CEIL_DIV(K, BLOCK_DIM);
    for (uint32_t t = 0; t < numTiles; ++t) {
        uint32_t tile_A_col = t * BLOCK_DIM + threadIdx.x;
        uint32_t tile_B_row = t * BLOCK_DIM + threadIdx.y;

        // Unified global coordinate boundary load into Shared Memory
        As[threadIdx.y][threadIdx.x] = (row < static_cast<uint32_t>(M) && tile_A_col < static_cast<uint32_t>(K)) ? A[row * K + tile_A_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (tile_B_row < static_cast<uint32_t>(K) && col < static_cast<uint32_t>(N)) ? B[tile_B_row * N + col] : 0.0f;

        __syncthreads();

        // Accumulate products from shared memory
        #pragma unroll
        for (uint32_t k = 0; k < BLOCK_DIM; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < static_cast<uint32_t>(M) && col < static_cast<uint32_t>(N)) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void run_sgemm_02_shared_mem(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));

    sgemm_02_shared_mem_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
