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

namespace {
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 128;
    constexpr uint32_t BK = 16;
    constexpr uint32_t TM = 8;
    constexpr uint32_t TN = 8;

    constexpr uint32_t THREADS_X = BN / TN; // 16
    constexpr uint32_t THREADS_Y = BM / TM; // 16
    constexpr uint32_t TOTAL_THREADS = THREADS_X * THREADS_Y; // 256

    constexpr uint32_t THREADS_K_A = BK / VEC_SIZE; // 16 / 4 = 4
    constexpr uint32_t ROWS_PER_LOAD_A = TOTAL_THREADS / THREADS_K_A; // 64

    constexpr uint32_t THREADS_N_B = BN / VEC_SIZE; // 128 / 4 = 32
    constexpr uint32_t ROWS_PER_LOAD_B = TOTAL_THREADS / THREADS_N_B; // 8

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert(BK % VEC_SIZE == 0, "BK must be divisible by VEC_SIZE");
}

__global__ void sgemm_04_2d_block_tiling_kernel(int M, int N, int K, float alpha,
                                                const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float beta,
                                                float* __restrict__ C) {
    uint32_t blockRow = blockIdx.y;
    uint32_t blockCol = blockIdx.x;

    uint32_t threadRow = threadIdx.y; // 0..15
    uint32_t threadCol = threadIdx.x; // 0..15

    __shared__ float As[BM][BK]; // 128 x 16
    __shared__ float Bs[BK][BN]; // 16 x 128

    // Registers for 2D micro-tile accumulation
    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255 threads

    // Threads collaboratively load As (128x16 = 2048 floats -> 8 floats per thread)
    uint32_t loadA_row0 = tid / THREADS_K_A;
    uint32_t loadA_row1 = loadA_row0 + ROWS_PER_LOAD_A;
    uint32_t loadA_col_start = (tid % THREADS_K_A) * VEC_SIZE;

    // Threads collaboratively load Bs (16x128 = 2048 floats -> 8 floats per thread)
    uint32_t loadB_row0 = tid / THREADS_N_B;
    uint32_t loadB_row1 = loadB_row0 + ROWS_PER_LOAD_B;
    uint32_t loadB_col_start = (tid % THREADS_N_B) * VEC_SIZE;

    for (int bk = 0; bk < K; bk += BK) {
        // Load tile from A into shared memory (2 chunks of 4 floats)
        #pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; ++i) {
            uint32_t cur_loadA_col = loadA_col_start + i;
            uint32_t gRowA0 = blockRow * BM + loadA_row0;
            uint32_t gRowA1 = blockRow * BM + loadA_row1;
            int gColA = bk + cur_loadA_col;
            As[loadA_row0][cur_loadA_col] = (gRowA0 < static_cast<uint32_t>(M) && gColA < K) ? A[gRowA0 * K + gColA] : 0.0f;
            As[loadA_row1][cur_loadA_col] = (gRowA1 < static_cast<uint32_t>(M) && gColA < K) ? A[gRowA1 * K + gColA] : 0.0f;
        }

        // Load tile from B into shared memory (2 chunks of 4 floats)
        #pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; ++i) {
            uint32_t cur_loadB_col = loadB_col_start + i;
            int gRowB0 = bk + loadB_row0;
            int gRowB1 = bk + loadB_row1;
            uint32_t gColB = blockCol * BN + cur_loadB_col;
            Bs[loadB_row0][cur_loadB_col] = (gRowB0 < K && gColB < static_cast<uint32_t>(N)) ? B[gRowB0 * N + gColB] : 0.0f;
            Bs[loadB_row1][cur_loadB_col] = (gRowB1 < K && gColB < static_cast<uint32_t>(N)) ? B[gRowB1 * N + gColB] : 0.0f;
        }

        __syncthreads();

        // Outer product calculation over BK
        #pragma unroll
        for (uint32_t k = 0; k < BK; ++k) {
            // Read A slice for this thread into regA
            #pragma unroll
            for (uint32_t m = 0; m < TM; ++m) {
                regA[m] = As[threadRow * TM + m][k];
            }
            // Read B slice for this thread into regB
            #pragma unroll
            for (uint32_t n = 0; n < TN; ++n) {
                regB[n] = Bs[k][threadCol * TN + n];
            }
            // Rank-1 outer product update
            #pragma unroll
            for (uint32_t m = 0; m < TM; ++m) {
                #pragma unroll
                for (uint32_t n = 0; n < TN; ++n) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }

        __syncthreads();
    }

    // Write back 2D register tile to global matrix C
    #pragma unroll
    for (uint32_t m = 0; m < TM; ++m) {
        uint32_t r = blockRow * BM + threadRow * TM + m;
        #pragma unroll
        for (uint32_t n = 0; n < TN; ++n) {
            uint32_t c = blockCol * BN + threadCol * TN + n;
            if (r < static_cast<uint32_t>(M) && c < static_cast<uint32_t>(N)) {
                C[r * N + c] = alpha * regC[m][n] + beta * C[r * N + c];
            }
        }
    }
}

void run_sgemm_04_2d_block_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(THREADS_X, THREADS_Y); // (16, 16) = 256 threads
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    sgemm_04_2d_block_tiling_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
