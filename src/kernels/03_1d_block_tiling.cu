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

namespace {
    constexpr uint32_t BM = 64;
    constexpr uint32_t BN = 64;
    constexpr uint32_t BK = 8;
    constexpr uint32_t TM = 8;

    constexpr uint32_t THREADS_X = BN;       // 64
    constexpr uint32_t THREADS_Y = BM / TM;  // 8
    constexpr uint32_t TOTAL_THREADS = THREADS_X * THREADS_Y; // 512

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(TOTAL_THREADS == BM * BK, "Threads must match elements in As tile");
    static_assert(TOTAL_THREADS == BK * BN, "Threads must match elements in Bs tile");
}

__global__ void sgemm_03_1d_block_tiling_kernel(int M, int N, int K, float alpha,
                                                const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float beta,
                                                float* __restrict__ C) {
    uint32_t blockRow = blockIdx.y;
    uint32_t blockCol = blockIdx.x;

    uint32_t threadRow = threadIdx.y; // 0..7
    uint32_t threadCol = threadIdx.x; // 0..63

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Thread position in output matrix
    uint32_t row_start = blockRow * BM + threadRow * TM;
    uint32_t col = blockCol * BN + threadCol;

    // Registers to store TM accumulators for this thread
    float regC[TM] = {0.0f};

    // Calculate linear thread ID for collaborative loading of shared memory
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..511 threads

    // Load tiles into shared memory:
    // As is BM x BK (64 x 8 = 512 elements -> 1 element per thread)
    uint32_t loadA_row = tid / BK;
    uint32_t loadA_col = tid % BK;

    // Bs is BK x BN (8 x 64 = 512 elements -> 1 element per thread)
    uint32_t loadB_row = tid / BN;
    uint32_t loadB_col = tid % BN;

    for (int bk = 0; bk < K; bk += BK) {
        // Load As tile
        uint32_t gRowA = blockRow * BM + loadA_row;
        int gColA = bk + loadA_col;
        if (gRowA < static_cast<uint32_t>(M) && gColA < K) {
            As[loadA_row][loadA_col] = A[gRowA * K + gColA];
        } else {
            As[loadA_row][loadA_col] = 0.0f;
        }

        // Load Bs tile
        int gRowB = bk + loadB_row;
        uint32_t gColB = blockCol * BN + loadB_col;
        if (gRowB < K && gColB < static_cast<uint32_t>(N)) {
            Bs[loadB_row][loadB_col] = B[gRowB * N + gColB];
        } else {
            Bs[loadB_row][loadB_col] = 0.0f;
        }

        __syncthreads();

        // Compute 1D thread tile outer product
        #pragma unroll
        for (uint32_t k = 0; k < BK; ++k) {
            float regB = Bs[k][threadCol];
            #pragma unroll
            for (uint32_t m = 0; m < TM; ++m) {
                regC[m] += As[threadRow * TM + m][k] * regB;
            }
        }

        __syncthreads();
    }

    // Write back results to global memory
    #pragma unroll
    for (uint32_t m = 0; m < TM; ++m) {
        uint32_t r = row_start + m;
        if (r < static_cast<uint32_t>(M) && col < static_cast<uint32_t>(N)) {
            C[r * N + col] = alpha * regC[m] + beta * C[r * N + col];
        }
    }
}

void run_sgemm_03_1d_block_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(THREADS_X, THREADS_Y); // (64, 8) = 512 threads
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    sgemm_03_1d_block_tiling_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
