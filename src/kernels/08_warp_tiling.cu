#include "sgemm.h"
#include "utils.h"

/*
 * Step 8: Hierarchical Warp Tiling (Block -> Warp -> Thread)
 * -----------------------------------------------------------------------------
 * Architecture:
 * - 3-Tier Hierarchical Decomposition:
 *   1. Block Tile:  BM = 128, BN = 128, BK = 16
 *   2. Warp Tile:   WM = 32,  WN = 64  (8 warps per block: 4 warps along M, 2 warps along N)
 *   3. Thread Tile: TM = 8,   TN = 8   (32 threads per warp: 4 threads along M, 8 threads along N)
 *
 * Benefits:
 * - Restructures shared memory load patterns per warp:
 *   - Each warp reads a compact WM x WN slice (32x64) from shared memory.
 *   - All 8 threads in a warp column read identical A slices -> 100% hardware broadcast.
 *   - All 4 threads in a warp row read identical B slices -> 100% hardware broadcast.
 *   - Warp column threads access only 8 distinct columns (rather than 16), eliminating
 *     high-degree bank conflicts on Matrix B.
 * - Double-Buffered Shared Memory (Level 1 Software Pipelining) with transposed A.
 * - Vectorized memory access across all stages: LDG.128, STS.128, LDS.128, and STG.128.
 */

namespace {
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 128;
    constexpr uint32_t BK = 16;

    // Warp hierarchy configuration
    constexpr uint32_t WARPS_M = 4; // 4 warps along M
    constexpr uint32_t WARPS_N = 2; // 2 warps along N

    constexpr uint32_t WM = BM / WARPS_M; // 32
    constexpr uint32_t WN = BN / WARPS_N; // 64

    // Warp thread arrangement (32 threads = 4x8)
    constexpr uint32_t THREADS_PER_WARP_M = 4;
    constexpr uint32_t THREADS_PER_WARP_N = 8;

    constexpr uint32_t TM = WM / THREADS_PER_WARP_M; // 8
    constexpr uint32_t TN = WN / THREADS_PER_WARP_N; // 8

    constexpr uint32_t PAD_A = 4;

    constexpr uint32_t TOTAL_WARPS = WARPS_M * WARPS_N; // 8
    constexpr uint32_t TOTAL_THREADS = TOTAL_WARPS * WARP_SIZE; // 256

    constexpr uint32_t THREADS_K_A = BK / VEC_SIZE; // 16 / 4 = 4 threads per row of A
    constexpr uint32_t ROWS_PER_LOAD_A = TOTAL_THREADS / THREADS_K_A; // 64 rows per pass

    constexpr uint32_t THREADS_N_B = BN / VEC_SIZE; // 128 / 4 = 32 threads per row of B
    constexpr uint32_t ROWS_PER_LOAD_B = TOTAL_THREADS / THREADS_N_B; // 8 rows per pass

    static_assert(THREADS_PER_WARP_M * THREADS_PER_WARP_N == WARP_SIZE, "Warp arrangement must match WARP_SIZE");
    static_assert(BM % WARPS_M == 0, "BM must be divisible by WARPS_M");
    static_assert(BN % WARPS_N == 0, "BN must be divisible by WARPS_N");
    static_assert(WM % THREADS_PER_WARP_M == 0, "WM must be divisible by THREADS_PER_WARP_M");
    static_assert(WN % THREADS_PER_WARP_N == 0, "WN must be divisible by THREADS_PER_WARP_N");
    static_assert(BK % VEC_SIZE == 0, "BK must be a multiple of VEC_SIZE");
    static_assert(BN % VEC_SIZE == 0, "BN must be a multiple of VEC_SIZE");
    static_assert(TM % VEC_SIZE == 0, "TM must be a multiple of VEC_SIZE");
    static_assert(TN % VEC_SIZE == 0, "TN must be a multiple of VEC_SIZE");
}

__global__ void sgemm_08_warp_tiling_kernel(int M, int N, int K, float alpha,
                                            const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float beta,
                                            float* __restrict__ C) {
    uint32_t blockRow = blockIdx.y;
    uint32_t blockCol = blockIdx.x;

    uint32_t lane_id = threadIdx.x;          // 0..31 (lane in warp)
    uint32_t warp_id = threadIdx.y;          // 0..7  (warp index in block)
    uint32_t tid = (warp_id * WARP_SIZE) + lane_id; // 0..255

    // Warp hierarchy decomposition
    uint32_t warp_row = warp_id / WARPS_N; // 0..3
    uint32_t warp_col = warp_id % WARPS_N; // 0..1

    uint32_t lane_row = lane_id / THREADS_PER_WARP_N; // 0..3
    uint32_t lane_col = lane_id % THREADS_PER_WARP_N; // 0..7

    // Offset of this thread's 8x8 micro-tile within the 128x128 block tile
    uint32_t thread_m_offset = warp_row * WM + lane_row * TM; // warp_row * 32 + lane_row * 8
    uint32_t thread_n_offset = warp_col * WN + lane_col * TN; // warp_col * 64 + lane_col * 8

    // Shared memory double buffers:
    // As[2][BK][BM + PAD_A] -> As is stored transposed: As[k][m]
    // Bs[2][BK][BN]         -> Bs is stored row-major: Bs[k][n]
    __shared__ float As[2][BK][BM + PAD_A]; // 2 x 16 x 132
    __shared__ float Bs[2][BK][BN];         // 2 x 16 x 128

    // Accumulators for 8x8 micro-tile (64 floats in registers)
    float regC[TM][TN] = {0.0f};

    // Staging registers for inner-loop compute
    float regA[TM];
    float regB[TN];

    // Global memory prefetch staging registers
    float4 prefetchA[2];
    float4 prefetchB[2];

    uint32_t loadA_row0 = tid / THREADS_K_A;
    uint32_t loadA_row1 = loadA_row0 + ROWS_PER_LOAD_A;
    uint32_t loadA_col  = (tid % THREADS_K_A) * VEC_SIZE;

    uint32_t loadB_row0 = tid / THREADS_N_B;
    uint32_t loadB_row1 = loadB_row0 + ROWS_PER_LOAD_B;
    uint32_t loadB_col  = (tid % THREADS_N_B) * VEC_SIZE;

    // Helper lambdas for fetching from global memory (LDG.128)
    auto fetch_A = [&](int bk, float4 val[2]) {
        uint32_t gRowA0 = blockRow * BM + loadA_row0;
        uint32_t gRowA1 = blockRow * BM + loadA_row1;
        int gColA = bk + loadA_col;
        val[0] = *reinterpret_cast<const float4*>(&A[gRowA0 * K + gColA]);
        val[1] = *reinterpret_cast<const float4*>(&A[gRowA1 * K + gColA]);
    };

    auto fetch_B = [&](int bk, float4 val[2]) {
        int gRowB0 = bk + loadB_row0;
        int gRowB1 = bk + loadB_row1;
        uint32_t gColB = blockCol * BN + loadB_col;
        val[0] = *reinterpret_cast<const float4*>(&B[gRowB0 * N + gColB]);
        val[1] = *reinterpret_cast<const float4*>(&B[gRowB1 * N + gColB]);
    };

    // Helper lambda to commit fetched data into Shared Memory buffer write_idx
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

    // Main Pipelined Loop over K dimension
    for (int bk = BK; bk < K; bk += BK) {
        // Level 1 Prefetch: Global Memory -> Registers
        fetch_A(bk, prefetchA);
        fetch_B(bk, prefetchB);

        // Inner compute loop for the current tile in SMEM
        #pragma unroll
        for (uint32_t k = 0; k < BK; ++k) {
            // Warp-tiled Vectorized LDS.128 for A (contiguous along M due to transpose)
            *reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + 0]);
            *reinterpret_cast<float4*>(&regA[4]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + 4]);

            // Warp-tiled Vectorized LDS.128 for B
            *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + 0]);
            *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + 4]);

            // Rank-1 outer product update on regC
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

        // Swap ping-pong SMEM buffer indices
        read_idx ^= 1;
        write_idx ^= 1;
    }

    // Epilogue: Process the final tile in shared memory
    #pragma unroll
    for (uint32_t k = 0; k < BK; ++k) {
        *reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + 0]);
        *reinterpret_cast<float4*>(&regA[4]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + 4]);

        *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + 0]);
        *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + 4]);

        #pragma unroll
        for (uint32_t m = 0; m < TM; ++m) {
            #pragma unroll
            for (uint32_t n = 0; n < TN; ++n) {
                regC[m][n] += regA[m] * regB[n];
            }
        }
    }

    // Vectorized write back to matrix C (STG.128)
    #pragma unroll
    for (uint32_t m = 0; m < TM; ++m) {
        uint32_t r = blockRow * BM + thread_m_offset + m;

        #pragma unroll
        for (uint32_t n = 0; n < TN; n += 4) {
            uint32_t c = blockCol * BN + thread_n_offset + n;
            float4 oldC = *reinterpret_cast<const float4*>(&C[r * N + c]);
            float4 c_reg = *reinterpret_cast<const float4*>(&regC[m][n]);
            *reinterpret_cast<float4*>(&C[r * N + c]) = alpha * c_reg + beta * oldC;
        }
    }
}

void run_sgemm_08_warp_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(WARP_SIZE, TOTAL_WARPS); // 32 threads per warp x 8 warps = 256 threads
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    sgemm_08_warp_tiling_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
