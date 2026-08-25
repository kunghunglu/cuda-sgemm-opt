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

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert((BM * BK) % TOTAL_THREADS == 0, "As elements must be divisible by TOTAL_THREADS");
    static_assert((BK * BN) % TOTAL_THREADS == 0, "Bs elements must be divisible by TOTAL_THREADS");
}

__global__ void sgemm_04_2d_block_tiling_kernel(int M, int N, int K, float alpha,
                                                const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float beta,
                                                float* __restrict__ C) {
    const uint32_t uM = static_cast<uint32_t>(M);
    const uint32_t uN = static_cast<uint32_t>(N);
    const uint32_t uK = static_cast<uint32_t>(K);

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

    for (uint32_t bk = 0; bk < uK; bk += BK) {
        // Load tile from A into shared memory with contiguous scalar coalescing
        #pragma unroll
        for (uint32_t offset = 0; offset < BM * BK; offset += TOTAL_THREADS) {
            uint32_t idx = tid + offset;
            uint32_t rowA = idx / BK;
            uint32_t colA = idx % BK;
            uint32_t gRowA = blockRow * BM + rowA;
            uint32_t gColA = bk + colA;
            As[rowA][colA] = (gRowA < uM && gColA < uK) ? A[gRowA * uK + gColA] : 0.0f;
        }

        // Load tile from B into shared memory with contiguous scalar coalescing
        #pragma unroll
        for (uint32_t offset = 0; offset < BK * BN; offset += TOTAL_THREADS) {
            uint32_t idx = tid + offset;
            uint32_t rowB = idx / BN;
            uint32_t colB = idx % BN;
            uint32_t gRowB = bk + rowB;
            uint32_t gColB = blockCol * BN + colB;
            Bs[rowB][colB] = (gRowB < uK && gColB < uN) ? B[gRowB * uN + gColB] : 0.0f;
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
            if (r < uM && c < uN) {
                C[r * uN + c] = alpha * regC[m][n] + beta * C[r * uN + c];
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
