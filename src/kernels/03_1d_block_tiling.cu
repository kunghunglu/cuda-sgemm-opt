#include "sgemm.h"
#include "utils.h"

/*
 * Step 3: 1D Thread Tiling
 * ------------------------
 * Concept:
 * - Increases arithmetic intensity by assigning TM (e.g. 8) elements of C to each thread.
 * - Block tile size: BM x BN x BK = 64 x 64 x 8.
 * - Thread block configuration: (BN, BM / TM) = (64, 8) -> 512 threads per block.
 * - Each thread computes a vertical column slice of 8 elements in C.
 * - For each k step in BK, the thread loads 1 element of B into a register and reuses it
 *   across TM=8 inner products with A elements.
 * - Reduces shared memory accesses for matrix B by a factor of TM.
 */

#define BM_STEP3 64
#define BN_STEP3 64
#define BK_STEP3 8
#define TM_STEP3 8

__global__ void sgemm_03_1d_block_tiling_kernel(int M, int N, int K, float alpha,
                                                const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float beta,
                                                float* __restrict__ C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y; // 0..7
    int threadCol = threadIdx.x; // 0..63

    __shared__ float As[BM_STEP3][BK_STEP3];
    __shared__ float Bs[BK_STEP3][BN_STEP3];

    // Thread position in output matrix
    int row_start = blockRow * BM_STEP3 + threadRow * TM_STEP3;
    int col = blockCol * BN_STEP3 + threadCol;

    // Registers to store TM accumulators for this thread
    float regC[TM_STEP3] = {0.0f};

    // Calculate linear thread ID for collaborative loading of shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..511 threads

    // Load tiles into shared memory:
    // As is BM x BK (64 x 8 = 512 elements -> 1 element per thread)
    int loadA_row = tid / BK_STEP3;
    int loadA_col = tid % BK_STEP3;

    // Bs is BK x BN (8 x 64 = 512 elements -> 1 element per thread)
    int loadB_row = tid / BN_STEP3;
    int loadB_col = tid % BN_STEP3;

    for (int bk = 0; bk < K; bk += BK_STEP3) {
        // Load As tile
        int gRowA = blockRow * BM_STEP3 + loadA_row;
        int gColA = bk + loadA_col;
        if (gRowA < M && gColA < K) {
            As[loadA_row][loadA_col] = A[gRowA * K + gColA];
        } else {
            As[loadA_row][loadA_col] = 0.0f;
        }

        // Load Bs tile
        int gRowB = bk + loadB_row;
        int gColB = blockCol * BN_STEP3 + loadB_col;
        if (gRowB < K && gColB < N) {
            Bs[loadB_row][loadB_col] = B[gRowB * N + gColB];
        } else {
            Bs[loadB_row][loadB_col] = 0.0f;
        }

        __syncthreads();

        // Compute 1D thread tile outer product
        #pragma unroll
        for (int k = 0; k < BK_STEP3; ++k) {
            float regB = Bs[k][threadCol];
            #pragma unroll
            for (int m = 0; m < TM_STEP3; ++m) {
                regC[m] += As[threadRow * TM_STEP3 + m][k] * regB;
            }
        }

        __syncthreads();
    }

    // Write back results to global memory
    #pragma unroll
    for (int m = 0; m < TM_STEP3; ++m) {
        int r = row_start + m;
        if (r < M && col < N) {
            C[r * N + col] = alpha * regC[m] + beta * C[r * N + col];
        }
    }
}

void run_sgemm_03_1d_block_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(BN_STEP3, BM_STEP3 / TM_STEP3); // (64, 8) = 512 threads
    dim3 grid(CEIL_DIV(N, BN_STEP3), CEIL_DIV(M, BM_STEP3));

    sgemm_03_1d_block_tiling_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
