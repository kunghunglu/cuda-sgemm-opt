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

#define BM_STEP5 128
#define BN_STEP5 128
#define BK_STEP5 8
#define TM_STEP5 8
#define TN_STEP5 8

__device__ inline float4 operator*(float a, float4 b) {
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__device__ inline float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void sgemm_05_vectorized_kernel(int M, int N, int K, float alpha,
                                           const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float beta,
                                           float* __restrict__ C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y; // 0..15
    int threadCol = threadIdx.x; // 0..15

    __shared__ float As[BM_STEP5][BK_STEP5]; // 128 x 8
    __shared__ float Bs[BK_STEP5][BN_STEP5]; // 8 x 128

    float regC[TM_STEP5][TN_STEP5] = {0.0f};
    float regA[TM_STEP5];
    float regB[TN_STEP5];

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255

    // A tile loading: 128x8 = 1024 floats = 256 float4s -> 1 float4 per thread
    // Thread tid loads row = tid / 2, col = (tid % 2) * 4
    int loadA_row = tid / (BK_STEP5 / 4);
    int loadA_col = (tid % (BK_STEP5 / 4)) * 4;

    // B tile loading: 8x128 = 1024 floats = 256 float4s -> 1 float4 per thread
    // Thread tid loads row = tid / 32, col = (tid % 32) * 4
    int loadB_row = tid / (BN_STEP5 / 4);
    int loadB_col = (tid % (BN_STEP5 / 4)) * 4;

    for (int bk = 0; bk < K; bk += BK_STEP5) {
        // Vectorized load float4 from A
        int gRowA = blockRow * BM_STEP5 + loadA_row;
        int gColA = bk + loadA_col;
        *reinterpret_cast<float4*>(&As[loadA_row][loadA_col]) = *reinterpret_cast<const float4*>(&A[gRowA * K + gColA]);

        // Vectorized load float4 from B
        int gRowB = bk + loadB_row;
        int gColB = blockCol * BN_STEP5 + loadB_col;
        *reinterpret_cast<float4*>(&Bs[loadB_row][loadB_col]) = *reinterpret_cast<const float4*>(&B[gRowB * N + gColB]);

        __syncthreads();

        // Perform compute on current tile
        #pragma unroll
        for (int k = 0; k < BK_STEP5; ++k) {
            #pragma unroll
            for (int m = 0; m < TM_STEP5; ++m) {
                regA[m] = As[threadRow * TM_STEP5 + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN_STEP5; ++n) {
                regB[n] = Bs[k][threadCol * TN_STEP5 + n];
            }
            #pragma unroll
            for (int m = 0; m < TM_STEP5; ++m) {
                #pragma unroll
                for (int n = 0; n < TN_STEP5; ++n) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }

        __syncthreads();
    }

    // Vectorized store float4 back to C
    #pragma unroll
    for (int m = 0; m < TM_STEP5; ++m) {
        int r = blockRow * BM_STEP5 + threadRow * TM_STEP5 + m;

        #pragma unroll
        for (int n = 0; n < TN_STEP5; n += 4) {
            int c = blockCol * BN_STEP5 + threadCol * TN_STEP5 + n;
            float4 oldC = *reinterpret_cast<const float4*>(&C[r * N + c]);
            float4 c_reg = *reinterpret_cast<const float4*>(&regC[m][n]);
            *reinterpret_cast<float4*>(&C[r * N + c]) = alpha * c_reg + beta * oldC;
    
        }
    }
}

void run_sgemm_05_vectorized(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    dim3 block(BN_STEP5 / TN_STEP5, BM_STEP5 / TM_STEP5); // (16, 16)
    dim3 grid(CEIL_DIV(N, BN_STEP5), CEIL_DIV(M, BM_STEP5)); 

    sgemm_05_vectorized_kernel<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
}
