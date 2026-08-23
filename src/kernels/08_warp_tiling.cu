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

#define BM_STEP8 128
#define BN_STEP8 128
#define BK_STEP8 16

// Warp hierarchy configuration
#define WARPS_M_STEP8 4 // 4 warps along M
#define WARPS_N_STEP8 2 // 2 warps along N

#define WM_STEP8 (BM_STEP8 / WARPS_M_STEP8) // 32
#define WN_STEP8 (BN_STEP8 / WARPS_N_STEP8) // 64

// Warp thread arrangement (32 threads = 4x8)
#define THREADS_PER_WARP_M_STEP8 4
#define THREADS_PER_WARP_N_STEP8 8

#define TM_STEP8 (WM_STEP8 / THREADS_PER_WARP_M_STEP8) // 8
#define TN_STEP8 (WN_STEP8 / THREADS_PER_WARP_N_STEP8) // 8

#define PAD_A_STEP8 4

__global__ void sgemm_08_warp_tiling_kernel(int M, int N, int K, float alpha,
                                            const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float beta,
                                            float* __restrict__ C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    uint32_t lane_id = threadIdx.x;          // 0..31 (lane in warp)
    uint32_t warp_id = threadIdx.y;          // 0..7  (warp index in block)
    uint32_t tid = (warp_id * 32) + lane_id; // 0..255

    // Warp hierarchy decomposition
    uint32_t warp_row = warp_id / WARPS_N_STEP8; // 0..3
    uint32_t warp_col = warp_id % WARPS_N_STEP8; // 0..1

    uint32_t lane_row = lane_id / THREADS_PER_WARP_N_STEP8; // 0..3
    uint32_t lane_col = lane_id % THREADS_PER_WARP_N_STEP8; // 0..7

    // Offset of this thread's 8x8 micro-tile within the 128x128 block tile
    uint32_t thread_m_offset = warp_row * WM_STEP8 + lane_row * TM_STEP8; // warp_row * 32 + lane_row * 8
    uint32_t thread_n_offset = warp_col * WN_STEP8 + lane_col * TN_STEP8; // warp_col * 64 + lane_col * 8

    // Shared memory double buffers:
    // As[2][BK][BM + PAD_A] -> As is stored transposed: As[k][m]
    // Bs[2][BK][BN]         -> Bs is stored row-major: Bs[k][n]
    __shared__ float As[2][BK_STEP8][BM_STEP8 + PAD_A_STEP8]; // 2 x 16 x 132
    __shared__ float Bs[2][BK_STEP8][BN_STEP8];               // 2 x 16 x 128

    // Accumulators for 8x8 micro-tile (64 floats in registers)
    float regC[TM_STEP8][TN_STEP8] = {0.0f};

    // Staging registers for inner-loop compute
    float regA[TM_STEP8];
    float regB[TN_STEP8];

    // Global memory prefetch staging registers
    float4 prefetchA[2];
    float4 prefetchB[2];

    // Global memory tile loading mapping (256 threads load 2048 floats = 512 float4s -> 2 float4 per thread)
    constexpr uint32_t VEC_SIZE = 4;
    constexpr uint32_t THREADS_K_A = BK_STEP8 / VEC_SIZE; // 16 / 4 = 4 threads per row of A
    constexpr uint32_t ROWS_PER_LOAD_A = 256 / THREADS_K_A; // 64 rows per pass

    uint32_t loadA_row0 = tid / THREADS_K_A;
    uint32_t loadA_row1 = loadA_row0 + ROWS_PER_LOAD_A;
    uint32_t loadA_col  = (tid % THREADS_K_A) * VEC_SIZE;

    constexpr uint32_t THREADS_N_B = BN_STEP8 / VEC_SIZE; // 128 / 4 = 32 threads per row of B
    constexpr uint32_t ROWS_PER_LOAD_B = 256 / THREADS_N_B; // 8 rows per pass

    uint32_t loadB_row0 = tid / THREADS_N_B;
    uint32_t loadB_row1 = loadB_row0 + ROWS_PER_LOAD_B;
    uint32_t loadB_col  = (tid % THREADS_N_B) * VEC_SIZE;

    // Helper lambdas for fetching from global memory (LDG.128)
    auto fetch_A = [&](int bk, float4 val[2]) {
        int gRowA0 = blockRow * BM_STEP8 + loadA_row0;
        int gRowA1 = blockRow * BM_STEP8 + loadA_row1;
        int gColA = bk + loadA_col;
        val[0] = *reinterpret_cast<const float4*>(&A[gRowA0 * K + gColA]);
        val[1] = *reinterpret_cast<const float4*>(&A[gRowA1 * K + gColA]);
    };

    auto fetch_B = [&](int bk, float4 val[2]) {
        int gRowB0 = bk + loadB_row0;
        int gRowB1 = bk + loadB_row1;
        int gColB = blockCol * BN_STEP8 + loadB_col;
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
    for (int bk = BK_STEP8; bk < K; bk += BK_STEP8) {
        // Level 1 Prefetch: Global Memory -> Registers
        fetch_A(bk, prefetchA);
        fetch_B(bk, prefetchB);

        // Inner compute loop for the current tile in SMEM
        #pragma unroll
        for (int k = 0; k < BK_STEP8; ++k) {
            // Warp-tiled Vectorized LDS.128 for A (contiguous along M due to transpose)
            *reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + 0]);
            *reinterpret_cast<float4*>(&regA[4]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + 4]);

            // Warp-tiled Vectorized LDS.128 for B
            *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + 0]);
            *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + 4]);

            // Rank-1 outer product update on regC
            #pragma unroll
            for (int m = 0; m < TM_STEP8; ++m) {
                #pragma unroll
                for (int n = 0; n < TN_STEP8; ++n) {
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
    for (int k = 0; k < BK_STEP8; ++k) {
        *reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + 0]);
        *reinterpret_cast<float4*>(&regA[4]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + 4]);

        *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + 0]);
        *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + 4]);

        #pragma unroll
        for (int m = 0; m < TM_STEP8; ++m) {
            #pragma unroll
            for (int n = 0; n < TN_STEP8; ++n) {
                regC[m][n] += regA[m] * regB[n];
            }
        }
    }

    // Vectorized write back to matrix C (STG.128)
    #pragma unroll
    for (int m = 0; m < TM_STEP8; ++m) {
        int r = blockRow * BM_STEP8 + thread_m_offset + m;

        #pragma unroll
        for (int n = 0; n < TN_STEP8; n += 4) {
            int c = blockCol * BN_STEP8 + thread_n_offset + n;
            float4 oldC = *reinterpret_cast<const float4*>(&C[r * N + c]);
            float4 c_reg = *reinterpret_cast<const float4*>(&regC[m][n]);
            *reinterpret_cast<float4*>(&C[r * N + c]) = alpha * c_reg + beta * oldC;
        }
    }
}

void run_sgemm_08_warp_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(32, 8); // 32 threads per warp x 8 warps = 256 threads
    dim3 grid(CEIL_DIV(N, BN_STEP8), CEIL_DIV(M, BM_STEP8));

    sgemm_08_warp_tiling_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
