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
    static_assert(BK % VEC_SIZE == 0, "BK must be a multiple of VEC_SIZE");
    static_assert(BN % VEC_SIZE == 0, "BN must be a multiple of VEC_SIZE");
    static_assert(TN % VEC_SIZE == 0, "TN must be a multiple of VEC_SIZE");
}

__global__ void sgemm_06_smem_double_buffering_kernel(int M, int N, int K, float alpha,
                                                     const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float beta,
                                                     float* __restrict__ C) {
    uint32_t blockRow = blockIdx.y;
    uint32_t blockCol = blockIdx.x;

    uint32_t threadRow = threadIdx.y; // 0..15
    uint32_t threadCol = threadIdx.x; // 0..15

    // Double buffers in shared memory
    __shared__ float As[2][BM][BK]; // 2 x 128 x 16 = 16 KB
    __shared__ float Bs[2][BK][BN]; // 2 x 16 x 128 = 16 KB

    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    // Staging registers for double buffering / latency hiding:
    // 1. GMEM -> SMEM uses registers implicitly (pre-Ampere has no direct GMEM->SMEM instruction; LDG loads to registers first).
    // 2. Explicitly naming prefetch registers decouples the load (LDG) from the store (STS), overlapping DRAM fetch with ALU math.
    // 3. We must commit to SMEM because registers are thread-private (cannot be shared across threads) and limited in size (max 255);
    //    SMEM enables 256 threads to share/broadcast tile data, saving 16x DRAM bandwidth without register spilling.
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
    for (int bk = BK; bk < K; bk += BK) {
        // Prefetch next tile from global memory into registers
        fetch_A(bk, prefetchA);
        fetch_B(bk, prefetchB);

        // Compute on current read_idx shared memory buffer
        #pragma unroll
        for (uint32_t k = 0; k < BK; ++k) {
            #pragma unroll
            for (uint32_t m = 0; m < TM; ++m) {
                regA[m] = As[read_idx][threadRow * TM + m][k];
            }
            // Vectorized LDS.128: Load 8 contiguous floats of B from Shared Memory
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
    for (uint32_t k = 0; k < BK; ++k) {
        #pragma unroll
        for (uint32_t m = 0; m < TM; ++m) {
            regA[m] = As[read_idx][threadRow * TM + m][k];
        }
        // Vectorized LDS.128: Load 8 contiguous floats of B from Shared Memory
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

    // Write back results
    #pragma unroll
    for (uint32_t m = 0; m < TM; ++m) {
        uint32_t r = blockRow * BM + threadRow * TM + m;

        #pragma unroll
        for (uint32_t n = 0; n < TN; n += 4) {
            uint32_t c = blockCol * BN + threadCol * TN + n;
            float4 oldC = *reinterpret_cast<const float4*>(&C[r * N + c]);
            float4 c_reg = *reinterpret_cast<const float4*>(&regC[m][n]);
            *reinterpret_cast<float4*>(&C[r * N + c]) = alpha * c_reg + beta * oldC;
        }
    }
}

void run_sgemm_06_smem_double_buffering(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(THREADS_X, THREADS_Y); // (16, 16)
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    sgemm_06_smem_double_buffering_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
