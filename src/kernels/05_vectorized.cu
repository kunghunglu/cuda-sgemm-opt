#include "sgemm.h"
#include "utils.h"

/*
 * Step 5: Vectorized Memory Access (float4)
 * ----------------------------------------
 * Concept:
 * - Replaces 32-bit scalar memory instructions with 128-bit vectorized vector instructions (float4).
 * - Issues `LDG.128` (128-bit load) and `STG.128` (128-bit store) instructions to memory.
 * - Drastically reduces total instruction count and maximizes memory throughput.
 * - Requires matrix pointers and dimensions to be 16-byte aligned (multiples of 4 float elements).
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

__global__ void sgemm_05_vectorized_kernel(int M, int N, int K, float alpha,
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

    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255

    // A tile loading: 128x16 = 2048 floats = 512 float4s -> 2 float4 per thread
    uint32_t loadA_row0 = tid / THREADS_K_A;
    uint32_t loadA_row1 = loadA_row0 + ROWS_PER_LOAD_A;
    uint32_t loadA_col  = (tid % THREADS_K_A) * VEC_SIZE;

    // B tile loading: 16x128 = 2048 floats = 512 float4s -> 2 float4 per thread
    uint32_t loadB_row0 = tid / THREADS_N_B;
    uint32_t loadB_row1 = loadB_row0 + ROWS_PER_LOAD_B;
    uint32_t loadB_col  = (tid % THREADS_N_B) * VEC_SIZE;

    for (int bk = 0; bk < K; bk += BK) {
        // Vectorized load float4 from A
        uint32_t gRowA0 = blockRow * BM + loadA_row0;
        uint32_t gRowA1 = blockRow * BM + loadA_row1;
        int gColA = bk + loadA_col;
        *reinterpret_cast<float4*>(&As[loadA_row0][loadA_col]) = *reinterpret_cast<const float4*>(&A[gRowA0 * K + gColA]);
        *reinterpret_cast<float4*>(&As[loadA_row1][loadA_col]) = *reinterpret_cast<const float4*>(&A[gRowA1 * K + gColA]);

        // Vectorized load float4 from B
        int gRowB0 = bk + loadB_row0;
        int gRowB1 = bk + loadB_row1;
        uint32_t gColB = blockCol * BN + loadB_col;
        *reinterpret_cast<float4*>(&Bs[loadB_row0][loadB_col]) = *reinterpret_cast<const float4*>(&B[gRowB0 * N + gColB]);
        *reinterpret_cast<float4*>(&Bs[loadB_row1][loadB_col]) = *reinterpret_cast<const float4*>(&B[gRowB1 * N + gColB]);

        __syncthreads();

        // Perform compute on current tile
        #pragma unroll
        for (uint32_t k = 0; k < BK; ++k) {
            #pragma unroll
            for (uint32_t m = 0; m < TM; ++m) {
                regA[m] = As[threadRow * TM + m][k];
            }
            // Vectorized LDS.128: Load 8 contiguous floats of B from Shared Memory
            *reinterpret_cast<float4*>(&regB[0]) = *reinterpret_cast<const float4*>(&Bs[k][threadCol * TN + 0]);
            *reinterpret_cast<float4*>(&regB[4]) = *reinterpret_cast<const float4*>(&Bs[k][threadCol * TN + 4]);
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

    // Vectorized store float4 back to C
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

void run_sgemm_05_vectorized(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(THREADS_X, THREADS_Y); // (16, 16)
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); 

    sgemm_05_vectorized_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
