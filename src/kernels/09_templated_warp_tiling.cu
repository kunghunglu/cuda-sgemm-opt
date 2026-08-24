#include "sgemm.h"
#include "templated_warp_tiling.cuh"

/**
 * Kernel 9 Host Entry Point: Dispatches templated Kernel 9 with the tuned Golden Configuration.
 * 
 * Tuned Golden Configuration for Pascal / General Architectures:
 * BM = 128, BN = 128, BK = 16, WARPS_M = 2, WARPS_N = 4, TM = 8, TN = 8, PAD_A = 4
 */
void run_sgemm_09_templated_warp_tiling(int M, int N, int K, float alpha,
                                       const float* d_A, const float* d_B,
                                       float beta, float* d_C) {
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 128;
    constexpr uint32_t BK = 16;
    constexpr uint32_t WARPS_M = 2;
    constexpr uint32_t WARPS_N = 4;
    constexpr uint32_t TM = 8;
    constexpr uint32_t TN = 8;
    constexpr uint32_t PAD_A = 4;

    launch_templated_warp_tiling<BM, BN, BK, WARPS_M, WARPS_N, TM, TN, PAD_A>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );
}
