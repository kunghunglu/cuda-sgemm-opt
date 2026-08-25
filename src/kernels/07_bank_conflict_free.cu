#include "sgemm.h"
#include "utils.h"

/*
 * Step 7: Shared Memory Bank Conflict Reduction (Padding + Matrix A Transpose)
 * -----------------------------------------------------------------------------
 * Features:
 * - 2D Block Tiling: BM = 128, BN = 128, BK = 16, TM = 8, TN = 8 (256 threads/block: 16x16).
 * - Vectorized Global Memory Access: float4 loads (LDG.128) and stores (STG.128).
 * - Shared Memory Double Buffering (Level 1 Software Pipelining): As[2], Bs[2] to hide DRAM latency.
 * - Transposed Storage for Matrix A in Shared Memory: As[2][BK][BM + PAD_A]
 *   - Storing A transposed makes the M dimension contiguous in shared memory, allowing threads to load
 *     regA using 2x float4 (LDS.128) instead of 8x scalar LDS.32 (75% fewer LDS instructions).
 * - Shared Memory Row Padding (PAD_A = 4):
 *   - Row strides of As are padded with 4 floats to avoid bank conflicts across warps.
 * - Single-stage register buffers (regA[TM], regB[TN]) with #pragma unroll for automatic compiler pipelining.
 */

namespace {
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 128;
    constexpr uint32_t BK = 16;
    constexpr uint32_t TM = 8;
    constexpr uint32_t TN = 8;
    constexpr uint32_t PAD_A = 4;

    constexpr uint32_t THREADS_X = BN / TN; // 16
    constexpr uint32_t THREADS_Y = BM / TM; // 16
    constexpr uint32_t TOTAL_THREADS = THREADS_X * THREADS_Y; // 256

    constexpr uint32_t THREADS_K_A = BK / VEC_SIZE; // 16 / 4 = 4
    constexpr uint32_t ROWS_PER_LOAD_A = TOTAL_THREADS / THREADS_K_A; // 64

    constexpr uint32_t THREADS_N_B = BN / VEC_SIZE; // 128 / 4 = 32
    constexpr uint32_t ROWS_PER_LOAD_B = TOTAL_THREADS / THREADS_N_B; // 8

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert(BK % VEC_SIZE == 0, "BK must be a multiple of VEC_SIZE");
    static_assert(BN % VEC_SIZE == 0, "BN must be a multiple of VEC_SIZE");
    static_assert(TN % VEC_SIZE == 0, "TN must be a multiple of VEC_SIZE");
}

__global__ void sgemm_07_bank_conflict_free_kernel(int M, int N, int K, float alpha,
                                                   const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float beta,
                                                   float* __restrict__ C) {
    const uint32_t uN = static_cast<uint32_t>(N);
    const uint32_t uK = static_cast<uint32_t>(K);

    uint32_t blockRow = blockIdx.y;
    uint32_t blockCol = blockIdx.x;

    uint32_t threadRow = threadIdx.y; // 0..15
    uint32_t threadCol = threadIdx.x; // 0..15

    // Level 1: Double buffers in shared memory with transpose for A and row padding
    __shared__ float As[2][BK][BM + PAD_A]; // 2 x 16 x 132
    __shared__ float Bs[2][BK][BN];         // 2 x 16 x 128

    // Accumulators for 2D micro-tile (8x8 = 64 floats)
    float regC[TM][TN] = {0.0f};

    // Registers for inner loop computation
    float regA[TM];
    float regB[TN];

    // Level 1: Staging registers for global memory prefetch
    float4 prefetchA[2];
    float4 prefetchB[2];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255

    // A tile loading: 128x16 = 2048 floats = 512 float4s -> 2 float4 per thread
    uint32_t loadA_row0 = tid / THREADS_K_A; // tid / 4: 0..63
    uint32_t loadA_row1 = loadA_row0 + ROWS_PER_LOAD_A; // 64..127
    uint32_t loadA_col  = (tid % THREADS_K_A) * VEC_SIZE; // 0, 4, 8, 12

    // B tile loading: 16x128 = 2048 floats = 512 float4s -> 2 float4 per thread
    uint32_t loadB_row0 = tid / THREADS_N_B; // tid / 32: 0..7
    uint32_t loadB_row1 = loadB_row0 + ROWS_PER_LOAD_B; // 8..15
    uint32_t loadB_col  = (tid % THREADS_N_B) * VEC_SIZE; // 0, 4, 8, ..., 124

    // Helper lambdas for fetching from global memory
    auto fetch_A = [&](uint32_t bk, float4 val[2]) {
        uint32_t gRowA0 = blockRow * BM + loadA_row0;
        uint32_t gRowA1 = blockRow * BM + loadA_row1;
        uint32_t gColA = bk + loadA_col;
        val[0] = *reinterpret_cast<const float4*>(&A[gRowA0 * uK + gColA]);
        val[1] = *reinterpret_cast<const float4*>(&A[gRowA1 * uK + gColA]);
    };

    auto fetch_B = [&](uint32_t bk, float4 val[2]) {
        uint32_t gRowB0 = bk + loadB_row0;
        uint32_t gRowB1 = bk + loadB_row1;
        uint32_t gColB = blockCol * BN + loadB_col;
        val[0] = *reinterpret_cast<const float4*>(&B[gRowB0 * uN + gColB]);
        val[1] = *reinterpret_cast<const float4*>(&B[gRowB1 * uN + gColB]);
    };

    // Helper lambda to store prefetched A & B into Shared Memory buffer write_idx
    auto store_smem = [&](int write_idx, const float4 a_val[2], const float4 b_val[2]) {
        // Transpose store A into As[k][m]
        As[write_idx][loadA_col + 0][loadA_row0] = a_val[0].x;
        As[write_idx][loadA_col + 1][loadA_row0] = a_val[0].y;
        As[write_idx][loadA_col + 2][loadA_row0] = a_val[0].z;
        As[write_idx][loadA_col + 3][loadA_row0] = a_val[0].w;

        As[write_idx][loadA_col + 0][loadA_row1] = a_val[1].x;
        As[write_idx][loadA_col + 1][loadA_row1] = a_val[1].y;
        As[write_idx][loadA_col + 2][loadA_row1] = a_val[1].z;
        As[write_idx][loadA_col + 3][loadA_row1] = a_val[1].w;

        // Store B into Bs[k][n]
        *reinterpret_cast<float4*>(&Bs[write_idx][loadB_row0][loadB_col]) = b_val[0];
        *reinterpret_cast<float4*>(&Bs[write_idx][loadB_row1][loadB_col]) = b_val[1];
    };

    // Prologue: Load initial tile (bk = 0) from global memory into SMEM buffer 0
    fetch_A(0, prefetchA);
    fetch_B(0, prefetchB);
    store_smem(0, prefetchA, prefetchB);

    __syncthreads();

    int write_idx = 1;
    int read_idx = 0;

    // Main Pipelined Loop
    for (uint32_t bk = BK; bk < uK; bk += BK) {
        // Level 1 Prefetch: Global Memory -> Registers
        fetch_A(bk, prefetchA);
        fetch_B(bk, prefetchB);

        // Inner compute loop
        #pragma unroll
        for (uint32_t k = 0; k < BK; ++k) {
            // Vectorized LDS.128 for A (enabled by transposed layout)
            *reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<const float4*>(&As[read_idx][k][threadRow * TM + 0]);
            *reinterpret_cast<float4*>(&regA[4]) = *reinterpret_cast<const float4*>(&As[read_idx][k][threadRow * TM + 4]);

            // Vectorized LDS.128 for B
            *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][threadCol * TN + 0]);
            *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][threadCol * TN + 4]);

            // Compute rank-1 outer product update on regC
            #pragma unroll
            for (uint32_t m = 0; m < TM; ++m) {
                #pragma unroll
                for (uint32_t n = 0; n < TN; ++n) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }

        // Level 1 Commit: Write prefetched global data to SMEM buffer write_idx
        store_smem(write_idx, prefetchA, prefetchB);

        __syncthreads();

        // Swap shared memory buffer indices
        read_idx ^= 1;
        write_idx ^= 1;
    }

    // Epilogue: Process the final tile in shared memory
    #pragma unroll
    for (uint32_t k = 0; k < BK; ++k) {
        *reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<const float4*>(&As[read_idx][k][threadRow * TM + 0]);
        *reinterpret_cast<float4*>(&regA[4]) = *reinterpret_cast<const float4*>(&As[read_idx][k][threadRow * TM + 4]);

        *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][threadCol * TN + 0]);
        *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][threadCol * TN + 4]);

        #pragma unroll
        for (uint32_t m = 0; m < TM; ++m) {
            #pragma unroll
            for (uint32_t n = 0; n < TN; ++n) {
                regC[m][n] += regA[m] * regB[n];
            }
        }
    }

    // Write back results to global matrix C
    #pragma unroll
    for (uint32_t m = 0; m < TM; ++m) {
        uint32_t r = blockRow * BM + threadRow * TM + m;

        #pragma unroll
        for (uint32_t n = 0; n < TN; n += 4) {
            uint32_t c = blockCol * BN + threadCol * TN + n;
            float4 oldC = *reinterpret_cast<const float4*>(&C[r * uN + c]);
            float4 c_reg = *reinterpret_cast<const float4*>(&regC[m][n]);
            *reinterpret_cast<float4*>(&C[r * uN + c]) = alpha * c_reg + beta * oldC;
        }
    }
}

void run_sgemm_07_bank_conflict_free(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(THREADS_X, THREADS_Y); // (16, 16) = 256 threads
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    sgemm_07_bank_conflict_free_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
