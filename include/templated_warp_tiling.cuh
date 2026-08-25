#ifndef TEMPLATED_WARP_TILING_CUH
#define TEMPLATED_WARP_TILING_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include "utils.h"

/**
 * Generic Templated Hierarchical Warp Tiling SGEMM Kernel
 *
 * Parameters:
 * - BM, BN, BK: Block Tile Dimensions
 * - WARPS_M, WARPS_N: Warp arrangement in block tile
 * - TM, TN: Micro-tile dimensions per thread
 * - PAD_A: Shared memory row padding for transposed A
 */
template <
    uint32_t BM,
    uint32_t BN,
    uint32_t BK,
    uint32_t WARPS_M,
    uint32_t WARPS_N,
    uint32_t TM,
    uint32_t TN,
    uint32_t PAD_A = 4
>
__global__ void sgemm_templated_warp_tiling_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    const uint32_t uN = static_cast<uint32_t>(N);
    const uint32_t uK = static_cast<uint32_t>(K);

    constexpr uint32_t WM = BM / WARPS_M;
    constexpr uint32_t WN = BN / WARPS_N;
    constexpr uint32_t THREADS_PER_WARP_M = WM / TM;
    constexpr uint32_t THREADS_PER_WARP_N = WN / TN;

    constexpr uint32_t TOTAL_WARPS = WARPS_M * WARPS_N;
    constexpr uint32_t TOTAL_THREADS = TOTAL_WARPS * WARP_SIZE;

    constexpr uint32_t THREADS_K_A = BK / VEC_SIZE;
    constexpr uint32_t ROWS_PER_LOAD_A = TOTAL_THREADS / THREADS_K_A;

    constexpr uint32_t THREADS_N_B = BN / VEC_SIZE;
    constexpr uint32_t ROWS_PER_LOAD_B = TOTAL_THREADS / THREADS_N_B;

    // Static assertions for geometry and hardware validation
    static_assert(BM % WARPS_M == 0, "BM must be divisible by WARPS_M");
    static_assert(BN % WARPS_N == 0, "BN must be divisible by WARPS_N");
    static_assert(WM % TM == 0, "WM must be divisible by TM");
    static_assert(WN % TN == 0, "WN must be divisible by TN");
    static_assert(THREADS_PER_WARP_M * THREADS_PER_WARP_N == WARP_SIZE, "Threads per warp must equal WARP_SIZE (32)");
    static_assert(BK % VEC_SIZE == 0, "BK must be a multiple of VEC_SIZE");
    static_assert(BN % VEC_SIZE == 0, "BN must be a multiple of VEC_SIZE");
    static_assert(TM % VEC_SIZE == 0, "TM must be a multiple of VEC_SIZE (4)");
    static_assert(TN % VEC_SIZE == 0, "TN must be a multiple of VEC_SIZE (4)");
    static_assert((TOTAL_THREADS * VEC_SIZE) % BK == 0, "Global load A mapping constraint");
    static_assert((TOTAL_THREADS * VEC_SIZE) % BN == 0, "Global load B mapping constraint");
    static_assert((BM * BK) % (TOTAL_THREADS * VEC_SIZE) == 0, "A tile size must be multiple of total load width");
    static_assert((BK * BN) % (TOTAL_THREADS * VEC_SIZE) == 0, "B tile size must be multiple of total load width");

    uint32_t blockRow = blockIdx.y;
    uint32_t blockCol = blockIdx.x;

    uint32_t lane_id = threadIdx.x;
    uint32_t warp_id = threadIdx.y;
    uint32_t tid = (warp_id * WARP_SIZE) + lane_id;

    // Warp coordinates in Warp Grid
    uint32_t warp_row = warp_id / WARPS_N;
    uint32_t warp_col = warp_id % WARPS_N;

    // Thread coordinates inside the Warp
    uint32_t lane_row = lane_id / THREADS_PER_WARP_N;
    uint32_t lane_col = lane_id % THREADS_PER_WARP_N;

    // Thread offset within the Block Tile
    uint32_t thread_m_offset = warp_row * WM + lane_row * TM;
    uint32_t thread_n_offset = warp_col * WN + lane_col * TN;

    // Double-buffered Shared Memory with padding for transposed A
    __shared__ float As[2][BK][BM + PAD_A];
    __shared__ float Bs[2][BK][BN];

    // Thread accumulators & register staging buffers
    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    // Global memory prefetch buffers
    constexpr uint32_t LOADS_A = (BM * BK) / (TOTAL_THREADS * VEC_SIZE);
    constexpr uint32_t LOADS_B = (BK * BN) / (TOTAL_THREADS * VEC_SIZE);

    float4 prefetchA[LOADS_A];
    float4 prefetchB[LOADS_B];

    // Global load indices
    uint32_t loadA_row0 = tid / THREADS_K_A;
    uint32_t loadA_col  = (tid % THREADS_K_A) * VEC_SIZE;

    uint32_t loadB_row0 = tid / THREADS_N_B;
    uint32_t loadB_col  = (tid % THREADS_N_B) * VEC_SIZE;

    auto fetch_A = [&](uint32_t bk, float4 val[LOADS_A]) {
        #pragma unroll
        for (uint32_t i = 0; i < LOADS_A; ++i) {
            uint32_t gRowA = blockRow * BM + loadA_row0 + i * ROWS_PER_LOAD_A;
            uint32_t gColA = bk + loadA_col;
            val[i] = *reinterpret_cast<const float4*>(&A[gRowA * uK + gColA]);
        }
    };

    auto fetch_B = [&](uint32_t bk, float4 val[LOADS_B]) {
        #pragma unroll
        for (uint32_t i = 0; i < LOADS_B; ++i) {
            uint32_t gRowB = bk + loadB_row0 + i * ROWS_PER_LOAD_B;
            uint32_t gColB = blockCol * BN + loadB_col;
            val[i] = *reinterpret_cast<const float4*>(&B[gRowB * uN + gColB]);
        }
    };

    auto store_smem = [&](int write_idx, const float4 a_val[LOADS_A], const float4 b_val[LOADS_B]) {
        #pragma unroll
        for (uint32_t i = 0; i < LOADS_A; ++i) {
            uint32_t rowA = loadA_row0 + i * ROWS_PER_LOAD_A;
            As[write_idx][loadA_col + 0][rowA] = a_val[i].x;
            As[write_idx][loadA_col + 1][rowA] = a_val[i].y;
            As[write_idx][loadA_col + 2][rowA] = a_val[i].z;
            As[write_idx][loadA_col + 3][rowA] = a_val[i].w;
        }

        #pragma unroll
        for (uint32_t i = 0; i < LOADS_B; ++i) {
            uint32_t rowB = loadB_row0 + i * ROWS_PER_LOAD_B;
            *reinterpret_cast<float4*>(&Bs[write_idx][rowB][loadB_col]) = b_val[i];
        }
    };

    // Stage 0: Initial prefetch & populate Shared Memory buffer 0
    fetch_A(0, prefetchA);
    fetch_B(0, prefetchB);
    store_smem(0, prefetchA, prefetchB);

    __syncthreads();

    int write_idx = 1;
    int read_idx = 0;

    // Main K-loop (Double-buffered Software Pipelining)
    for (uint32_t bk = BK; bk < uK; bk += BK) {
        // Asynchronously prefetch next tile from Global Memory
        fetch_A(bk, prefetchA);
        fetch_B(bk, prefetchB);

        // Compute current tile from Shared Memory
        #pragma unroll
        for (uint32_t k = 0; k < BK; ++k) {
            // Load vector micro-slice of A into registers
            #pragma unroll
            for (uint32_t m = 0; m < TM; m += VEC_SIZE) {
                *reinterpret_cast<float4*>(&regA[m]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + m]);
            }

            // Load vector micro-slice of B into registers
            #pragma unroll
            for (uint32_t n = 0; n < TN; n += VEC_SIZE) {
                *reinterpret_cast<float4*>(&regB[n]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + n]);
            }

            // Compute TM x TN outer-product accumulation
            #pragma unroll
            for (uint32_t m = 0; m < TM; ++m) {
                #pragma unroll
                for (uint32_t n = 0; n < TN; ++n) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }

        // Store prefetched data into the alternate Shared Memory buffer
        store_smem(write_idx, prefetchA, prefetchB);
        __syncthreads();

        // Swap ping-pong double buffers
        read_idx ^= 1;
        write_idx ^= 1;
    }

    // Process the final tile
    #pragma unroll
    for (uint32_t k = 0; k < BK; ++k) {
        #pragma unroll
        for (uint32_t m = 0; m < TM; m += VEC_SIZE) {
            *reinterpret_cast<float4*>(&regA[m]) = *reinterpret_cast<const float4*>(&As[read_idx][k][thread_m_offset + m]);
        }
        #pragma unroll
        for (uint32_t n = 0; n < TN; n += VEC_SIZE) {
            *reinterpret_cast<float4*>(&regB[n]) = *reinterpret_cast<const float4*>(&Bs[read_idx][k][thread_n_offset + n]);
        }

        #pragma unroll
        for (uint32_t m = 0; m < TM; ++m) {
            #pragma unroll
            for (uint32_t n = 0; n < TN; ++n) {
                regC[m][n] += regA[m] * regB[n];
            }
        }
    }

    // Epilogue: Write TM x TN results back to Global Memory C (Vectorized STG.128)
    #pragma unroll
    for (uint32_t m = 0; m < TM; ++m) {
        uint32_t r = blockRow * BM + thread_m_offset + m;
        #pragma unroll
        for (uint32_t n = 0; n < TN; n += VEC_SIZE) {
            uint32_t c = blockCol * BN + thread_n_offset + n;
            float4 oldC = *reinterpret_cast<const float4*>(&C[r * uN + c]);
            float4 c_reg = *reinterpret_cast<const float4*>(&regC[m][n]);
            *reinterpret_cast<float4*>(&C[r * uN + c]) = alpha * c_reg + beta * oldC;
        }
    }
}

template <
    uint32_t BM,
    uint32_t BN,
    uint32_t BK,
    uint32_t WARPS_M,
    uint32_t WARPS_N,
    uint32_t TM,
    uint32_t TN,
    uint32_t PAD_A = 4
>
inline void launch_templated_warp_tiling(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    constexpr uint32_t TOTAL_WARPS = WARPS_M * WARPS_N;
    dim3 block(WARP_SIZE, TOTAL_WARPS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    sgemm_templated_warp_tiling_kernel<BM, BN, BK, WARPS_M, WARPS_N, TM, TN, PAD_A>
        <<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

#endif // TEMPLATED_WARP_TILING_CUH
