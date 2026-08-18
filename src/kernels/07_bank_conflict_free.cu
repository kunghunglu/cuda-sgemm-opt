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

#define BM_STEP7 128
#define BN_STEP7 128
#define BK_STEP7 16
#define TM_STEP7 8
#define TN_STEP7 8
#define PAD_STEP7 4

__global__ void sgemm_07_bank_conflict_free_kernel(int M, int N, int K, float alpha,
                                                   const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float beta,
                                                   float* __restrict__ C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y; // 0..15
    int threadCol = threadIdx.x; // 0..15

    // Level 1: Double buffers in shared memory with transpose for A and row padding
    __shared__ float As[2][BK_STEP7][BM_STEP7 + PAD_STEP7]; // 2 x 16 x 132
    __shared__ float Bs[2][BK_STEP7][BN_STEP7];             // 2 x 16 x 128

    // Accumulators for 2D micro-tile (8x8 = 64 floats)
    float regC[TM_STEP7][TN_STEP7] = {0.0f};

    // Registers for inner loop computation
    float regA[TM_STEP7];
    float regB[TN_STEP7];

    // Level 1: Staging registers for global memory prefetch
    float4 prefetchA[2];
    float4 prefetchB[2];

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255

    // A tile loading: 128x16 = 2048 floats = 512 float4s -> 2 float4 per thread
    int loadA_row0 = tid / (BK_STEP7 / 4);      // tid / 4: 0..63
    int loadA_row1 = loadA_row0 + 64;           // 64..127
    int loadA_col = (tid % (BK_STEP7 / 4)) * 4; // 0, 4, 8, 12

    // B tile loading: 16x128 = 2048 floats = 512 float4s -> 2 float4 per thread
    int loadB_row0 = tid / (BN_STEP7 / 4);      // tid / 32: 0..7
    int loadB_row1 = loadB_row0 + 8;            // 8..15
    int loadB_col = (tid % (BN_STEP7 / 4)) * 4; // 0, 4, 8, ..., 124

    // Helper lambdas for fetching from global memory
    auto fetch_A = [&](int bk, float4 val[2]) {
        int gRowA0 = blockRow * BM_STEP7 + loadA_row0;
        int gRowA1 = blockRow * BM_STEP7 + loadA_row1;
        int gColA = bk + loadA_col;
        val[0] = *reinterpret_cast<const float4*>(&A[gRowA0 * K + gColA]);
        val[1] = *reinterpret_cast<const float4*>(&A[gRowA1 * K + gColA]);
    };

    auto fetch_B = [&](int bk, float4 val[2]) {
        int gRowB0 = bk + loadB_row0;
        int gRowB1 = bk + loadB_row1;
        int gColB = blockCol * BN_STEP7 + loadB_col;
        val[0] = *reinterpret_cast<const float4*>(&B[gRowB0 * N + gColB]);
        val[1] = *reinterpret_cast<const float4*>(&B[gRowB1 * N + gColB]);
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
    for (int bk = BK_STEP7; bk < K; bk += BK_STEP7) {
        // Level 1 Prefetch: Global Memory -> Registers
        fetch_A(bk, prefetchA);
        fetch_B(bk, prefetchB);

        // Inner compute loop
        #pragma unroll
        for (int k = 0; k < BK_STEP7; ++k) {
            // Vectorized LDS.128 for A (enabled by transposed layout)
            *reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<const float4*>(&As[read_idx][k][threadRow * TM_STEP7 + 0]);
            *reinterpret_cast<float4*>(&regA[4]) = *reinterpret_cast<const float4*>(&As[read_idx][k][threadRow * TM_STEP7 + 4]);

            // Vectorized LDS.128 for B
            *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][threadCol * TN_STEP7 + 0]);
            *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][threadCol * TN_STEP7 + 4]);

            // Compute rank-1 outer product update on regC
            #pragma unroll
            for (int m = 0; m < TM_STEP7; ++m) {
                #pragma unroll
                for (int n = 0; n < TN_STEP7; ++n) {
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
    for (int k = 0; k < BK_STEP7; ++k) {
        *reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<const float4*>(&As[read_idx][k][threadRow * TM_STEP7 + 0]);
        *reinterpret_cast<float4*>(&regA[4]) = *reinterpret_cast<const float4*>(&As[read_idx][k][threadRow * TM_STEP7 + 4]);

        *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][threadCol * TN_STEP7 + 0]);
        *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][threadCol * TN_STEP7 + 4]);

        #pragma unroll
        for (int m = 0; m < TM_STEP7; ++m) {
            #pragma unroll
            for (int n = 0; n < TN_STEP7; ++n) {
                regC[m][n] += regA[m] * regB[n];
            }
        }
    }

    // Write back results to global matrix C
    #pragma unroll
    for (int m = 0; m < TM_STEP7; ++m) {
        int r = blockRow * BM_STEP7 + threadRow * TM_STEP7 + m;

        #pragma unroll
        for (int n = 0; n < TN_STEP7; n += 4) {
            int c = blockCol * BN_STEP7 + threadCol * TN_STEP7 + n;
            float4 oldC = *reinterpret_cast<const float4*>(&C[r * N + c]);
            float4 c_reg = *reinterpret_cast<const float4*>(&regC[m][n]);
            *reinterpret_cast<float4*>(&C[r * N + c]) = alpha * c_reg + beta * oldC;
        }
    }
}

void run_sgemm_07_bank_conflict_free(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(BN_STEP7 / TN_STEP7, BM_STEP7 / TM_STEP7); // (16, 16) = 256 threads
    dim3 grid(CEIL_DIV(N, BN_STEP7), CEIL_DIV(M, BM_STEP7));

    sgemm_07_bank_conflict_free_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
