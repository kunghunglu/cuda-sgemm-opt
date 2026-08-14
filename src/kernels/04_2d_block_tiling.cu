#include "sgemm.h"
#include "utils.h"

/*
 * Step 4: 2D Thread Tiling
 * ------------------------
 * Concept:
 * - Each thread computes a 2D tile (TM x TN = 8 x 8 = 64 elements) of matrix C.
 * - Block tile sizes: BM x BN x BK = 128 x 128 x 8.
 * - Thread block configuration: (BN / TN, BM / TM) = (16, 16) -> 256 threads.
 * - In the inner loop, each thread loads TM=8 values of A and TN=8 values of B into registers.
 * - Computes a rank-1 outer product update on the TM x TN register tile `regC[TM][TN]`.
 * - Ratio of compute (TM*TN FMAs) to shared memory loads (TM + TN floats) is (8*8) / (8+8) = 4.0 FMAs per load!
 */

#define BM_STEP4 128
#define BN_STEP4 128
#define BK_STEP4 8
#define TM_STEP4 8
#define TN_STEP4 8

__global__ void sgemm_04_2d_block_tiling_kernel(int M, int N, int K, float alpha,
                                                const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float beta,
                                                float* __restrict__ C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y; // 0..15
    int threadCol = threadIdx.x; // 0..15

    __shared__ float As[BM_STEP4][BK_STEP4]; // 128 x 8
    __shared__ float Bs[BK_STEP4][BN_STEP4]; // 8 x 128

    // Registers for 2D micro-tile accumulation
    float regC[TM_STEP4][TN_STEP4] = {0.0f};
    float regA[TM_STEP4];
    float regB[TN_STEP4];

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255 threads

    // Threads collaboratively load As (128x8 = 1024 floats -> 4 floats per thread)
    // Each thread loads 4 consecutive columns in the same row of As.
    // tid / 2 -> row (0..127), (tid % 2) * 4 -> column start (0 or 4).
    int loadA_row = (tid * 4) / BK_STEP4;
    int loadA_col_start = (tid * 4) % BK_STEP4;

    // Threads collaboratively load Bs (8x128 = 1024 floats -> 4 floats per thread)
    int loadB_row = tid / (BN_STEP4 / 4);
    int loadB_col_start = (tid % (BN_STEP4 / 4)) * 4;

    for (int bk = 0; bk < K; bk += BK_STEP4) {
        // Load tile from A into shared memory
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int cur_loadA_col = loadA_col_start + i;
            int gRowA = blockRow * BM_STEP4 + loadA_row;
            int gColA = bk + cur_loadA_col;
            if (gRowA < M && gColA < K) {
                As[loadA_row][cur_loadA_col] = A[gRowA * K + gColA];
            } else {
                As[loadA_row][cur_loadA_col] = 0.0f;
            }
        }

        // Load tile from B into shared memory
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int cur_loadB_col = loadB_col_start + i;
            int gRowB = bk + loadB_row;
            int gColB = blockCol * BN_STEP4 + cur_loadB_col;
            if (gRowB < K && gColB < N) {
                Bs[loadB_row][cur_loadB_col] = B[gRowB * N + gColB];
            } else {
                Bs[loadB_row][cur_loadB_col] = 0.0f;
            }
        }

        __syncthreads();

        // Outer product calculation over BK
        #pragma unroll
        for (int k = 0; k < BK_STEP4; ++k) {
            // Read A slice for this thread into regA
            #pragma unroll
            for (int m = 0; m < TM_STEP4; ++m) {
                regA[m] = As[threadRow * TM_STEP4 + m][k];
            }
            // Read B slice for this thread into regB
            #pragma unroll
            for (int n = 0; n < TN_STEP4; ++n) {
                regB[n] = Bs[k][threadCol * TN_STEP4 + n];
            }
            // Rank-1 outer product update
            #pragma unroll
            for (int m = 0; m < TM_STEP4; ++m) {
                #pragma unroll
                for (int n = 0; n < TN_STEP4; ++n) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }

        __syncthreads();
    }

    // Write back 2D register tile to global matrix C
    #pragma unroll
    for (int m = 0; m < TM_STEP4; ++m) {
        int r = blockRow * BM_STEP4 + threadRow * TM_STEP4 + m;
        #pragma unroll
        for (int n = 0; n < TN_STEP4; ++n) {
            int c = blockCol * BN_STEP4 + threadCol * TN_STEP4 + n;
            if (r < M && c < N) {
                C[r * N + c] = alpha * regC[m][n] + beta * C[r * N + c];
            }
        }
    }
}

void run_sgemm_04_2d_block_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(BN_STEP4 / TN_STEP4, BM_STEP4 / TM_STEP4); // (16, 16) = 256 threads
    dim3 grid(CEIL_DIV(N, BN_STEP4), CEIL_DIV(M, BM_STEP4));

    sgemm_04_2d_block_tiling_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
