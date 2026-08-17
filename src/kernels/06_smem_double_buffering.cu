#include "sgemm.h"
#include "utils.h"

/*
 * Step 6: Double Buffering / Software Pipelining
 * ----------------------------------------------
 * Concept:
 * - Allocates two sets of shared memory buffers (As[2][BM][BK] and Bs[2][BK][BN]).
 * - While the CUDA cores compute matrix multiplications on buffer `comp_idx`,
 *   threads prefetch the NEXT tile from global memory into registers, then write to buffer `load_idx`.
 * - Hides global memory fetch latency behind arithmetic computation.
 */

#define BM_STEP6 128
#define BN_STEP6 128
#define BK_STEP6 16
#define TM_STEP6 8
#define TN_STEP6 8

__global__ void sgemm_06_smem_double_buffering_kernel(int M, int N, int K, float alpha,
                                                 const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float beta,
                                                 float* __restrict__ C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y; // 0..15
    int threadCol = threadIdx.x; // 0..15

    // Double buffers in shared memory
    __shared__ float As[2][BM_STEP6][BK_STEP6]; // 2 x 128 x 16 = 16 KB
    __shared__ float Bs[2][BK_STEP6][BN_STEP6]; // 2 x 16 x 128 = 16 KB

    float regC[TM_STEP6][TN_STEP6] = {0.0f};
    float regA[TM_STEP6];
    float regB[TN_STEP6];

    // Staging registers for double buffering / latency hiding:
    // 1. GMEM -> SMEM uses registers implicitly (pre-Ampere has no direct GMEM->SMEM instruction; LDG loads to registers first).
    // 2. Explicitly naming prefetch registers decouples the load (LDG) from the store (STS), overlapping DRAM fetch with ALU math.
    // 3. We must commit to SMEM because registers are thread-private (cannot be shared across threads) and limited in size (max 255);
    //    SMEM enables 256 threads to share/broadcast tile data, saving 16x DRAM bandwidth without register spilling.
    float4 prefetchA[2];
    float4 prefetchB[2];

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255

    // A tile loading: 128x16 = 2048 floats = 512 float4s -> 2 float4 per thread
    int loadA_row0 = tid / (BK_STEP6 / 4); // tid / 4: 0..63
    int loadA_row1 = loadA_row0 + 64;      // 64..127
    int loadA_col = (tid % (BK_STEP6 / 4)) * 4; // 0, 4, 8, 12

    // B tile loading: 16x128 = 2048 floats = 512 float4s -> 2 float4 per thread
    int loadB_row0 = tid / (BN_STEP6 / 4); // tid / 32: 0..7
    int loadB_row1 = loadB_row0 + 8;       // 8..15
    int loadB_col = (tid % (BN_STEP6 / 4)) * 4; // 0, 4, 8, ..., 124

    // Helper lambdas for fetching from global memory
    auto fetch_A = [&](int bk, float4 val[2]) {
        int gRowA0 = blockRow * BM_STEP6 + loadA_row0;
        int gRowA1 = blockRow * BM_STEP6 + loadA_row1;
        int gColA = bk + loadA_col;
        val[0] = *reinterpret_cast<const float4*>(&A[gRowA0 * K + gColA]);
        val[1] = *reinterpret_cast<const float4*>(&A[gRowA1 * K + gColA]);
    };

    auto fetch_B = [&](int bk, float4 val[2]) {
        int gRowB0 = bk + loadB_row0;
        int gRowB1 = bk + loadB_row1;
        int gColB = blockCol * BN_STEP6 + loadB_col;
        val[0] = *reinterpret_cast<const float4*>(&B[gRowB0 * N + gColB]);
        val[1] = *reinterpret_cast<const float4*>(&B[gRowB1 * N + gColB]);
    };

    // Prologue: Load initial tile (bk = 0) into buffer 0
    fetch_A(0, prefetchA);
    fetch_B(0, prefetchB);

    *reinterpret_cast<float4*>(&As[0][loadA_row0][loadA_col]) = prefetchA[0];
    *reinterpret_cast<float4*>(&As[0][loadA_row1][loadA_col]) = prefetchA[1];
    *reinterpret_cast<float4*>(&Bs[0][loadB_row0][loadB_col]) = prefetchB[0];
    *reinterpret_cast<float4*>(&Bs[0][loadB_row1][loadB_col]) = prefetchB[1];

    __syncthreads();

    int write_idx = 1;
    int read_idx = 0;

    // Main loop: Pipelined execution
    for (int bk = BK_STEP6; bk < K; bk += BK_STEP6) {
        // Prefetch next tile from global memory into registers
        fetch_A(bk, prefetchA);
        fetch_B(bk, prefetchB);

        // Compute on current read_idx shared memory buffer
        #pragma unroll
        for (int k = 0; k < BK_STEP6; ++k) {
            #pragma unroll
            for (int m = 0; m < TM_STEP6; ++m) {
                regA[m] = As[read_idx][threadRow * TM_STEP6 + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN_STEP6; ++n) {
                regB[n] = Bs[read_idx][k][threadCol * TN_STEP6 + n];
            }
            #pragma unroll
            for (int m = 0; m < TM_STEP6; ++m) {
                #pragma unroll
                for (int n = 0; n < TN_STEP6; ++n) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }

        // Write prefetched tile to write_idx shared memory buffer
        *reinterpret_cast<float4*>(&As[write_idx][loadA_row0][loadA_col]) = prefetchA[0];
        *reinterpret_cast<float4*>(&As[write_idx][loadA_row1][loadA_col]) = prefetchA[1];
        *reinterpret_cast<float4*>(&Bs[write_idx][loadB_row0][loadB_col]) = prefetchB[0];
        *reinterpret_cast<float4*>(&Bs[write_idx][loadB_row1][loadB_col]) = prefetchB[1];

        __syncthreads();

        // Swap buffer indices
        read_idx ^= 1;
        write_idx ^= 1;
    }

    // Epilogue: Compute on final tile remaining in read_idx buffer
    #pragma unroll
    for (int k = 0; k < BK_STEP6; ++k) {
        #pragma unroll
        for (int m = 0; m < TM_STEP6; ++m) {
            regA[m] = As[read_idx][threadRow * TM_STEP6 + m][k];
        }
        #pragma unroll
        for (int n = 0; n < TN_STEP6; ++n) {
            regB[n] = Bs[read_idx][k][threadCol * TN_STEP6 + n];
        }
        #pragma unroll
        for (int m = 0; m < TM_STEP6; ++m) {
            #pragma unroll
            for (int n = 0; n < TN_STEP6; ++n) {
                regC[m][n] += regA[m] * regB[n];
            }
        }
    }

    // Write back results
    #pragma unroll
    for (int m = 0; m < TM_STEP6; ++m) {
        int r = blockRow * BM_STEP6 + threadRow * TM_STEP6 + m;

        #pragma unroll
        for (int n = 0; n < TN_STEP6; n += 4) {
            int c = blockCol * BN_STEP6 + threadCol * TN_STEP6 + n;
            float4 oldC = *reinterpret_cast<const float4*>(&C[r * N + c]);
            float4 c_reg = *reinterpret_cast<const float4*>(&regC[m][n]);
            *reinterpret_cast<float4*>(&C[r * N + c]) = alpha * c_reg + beta * oldC;
        }
    }
}

void run_sgemm_06_smem_double_buffering(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(BN_STEP6 / TN_STEP6, BM_STEP6 / TM_STEP6); // (16, 16)
    dim3 grid(CEIL_DIV(N, BN_STEP6), CEIL_DIV(M, BM_STEP6));

    sgemm_06_smem_double_buffering_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
