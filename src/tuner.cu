#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.h"
#include "templated_warp_tiling.cuh"

struct ConfigResult {
    std::string name;
    uint32_t BM, BN, BK;
    uint32_t WARPS_M, WARPS_N;
    uint32_t TM, TN, PAD_A;
    uint32_t num_threads;
    uint32_t smem_bytes;
    float time_ms;
    double gflops;
    float max_error;
    double pct_cublas;
    bool passed;
};

// Runner helper for benchmarking a specific configuration
template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WARPS_M, uint32_t WARPS_N, uint32_t TM, uint32_t TN, uint32_t PAD_A = 4>
ConfigResult evaluate_config(const std::string& name, int M, int N, int K,
                             const float* d_A, const float* d_B, float* d_C,
                             const float* h_C_ref, float* h_C_test,
                             int warmup_iters, int bench_iters, double cublas_gflops) {
    constexpr uint32_t TOTAL_WARPS = WARPS_M * WARPS_N;
    constexpr uint32_t TOTAL_THREADS = TOTAL_WARPS * WARP_SIZE;
    constexpr uint32_t SMEM_BYTES = 2 * (BK * (BM + PAD_A) + BK * BN) * sizeof(float);

    dim3 block(WARP_SIZE, TOTAL_WARPS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        sgemm_templated_warp_tiling_kernel<BM, BN, BK, WARPS_M, WARPS_N, TM, TN, PAD_A>
            <<<grid, block>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Numerical Verification
    CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * M * N));
    sgemm_templated_warp_tiling_kernel<BM, BN, BK, WARPS_M, WARPS_N, TM, TN, PAD_A>
        <<<grid, block>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaMemcpy(h_C_test, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    float max_err = calc_max_abs_error(h_C_ref, h_C_test, M, N);
    bool passed = (max_err < 1e-2f);

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        sgemm_templated_warp_tiling_kernel<BM, BN, BK, WARPS_M, WARPS_N, TM, TN, PAD_A>
            <<<grid, block>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= bench_iters;

    double gflops = (2.0 * M * N * K * 1e-9) / (ms * 1e-3);
    double pct = (cublas_gflops > 0.0) ? (gflops / cublas_gflops * 100.0) : 0.0;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return {
        name,
        BM, BN, BK,
        WARPS_M, WARPS_N,
        TM, TN, PAD_A,
        TOTAL_THREADS,
        SMEM_BYTES,
        ms,
        gflops,
        max_err,
        pct,
        passed
    };
}

void run_autotune(int M, int N, int K, int warmup_iters, int bench_iters) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "=========================================================================================\n";
    std::cout << " CUDA SGEMM Kernel 9 Auto-Tuning Engine\n";
    std::cout << " Device: " << prop.name << " (" << prop.multiProcessorCount << " SMs, CC " << prop.major << "." << prop.minor << ")\n";
    std::cout << " Matrix Size: M=" << M << ", N=" << N << ", K=" << K << " | Total FLOPs: " 
              << std::scientific << std::setprecision(3) << (2.0 * M * N * K) << " FLOPs\n";
    std::cout << "=========================================================================================\n\n";

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C_ref = (float*)malloc(bytes_C);
    float* h_C_test = (float*)malloc(bytes_C);

    std::srand(42);
    randomize_matrix(h_A, M * K);
    randomize_matrix(h_B, K * N);
    zero_matrix(h_C_ref, M * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // cuBLAS Reference Benchmark
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmup_iters; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    cublas_ms /= bench_iters;
    double cublas_gflops = (2.0 * M * N * K * 1e-9) / (cublas_ms * 1e-3);
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, bytes_C, cudaMemcpyDeviceToHost));

    std::cout << " Reference cuBLAS Performance: " << std::fixed << std::setprecision(3) << cublas_ms 
              << " ms (" << std::setprecision(1) << cublas_gflops << " GFLOPS)\n\n";
    std::cout << " Exploring parameter search space...\n\n";

    std::vector<ConfigResult> results;

    #define EVAL(name, BM, BN, BK, WARPS_M, WARPS_N, TM, TN, PAD_A) \
        results.push_back(evaluate_config<BM, BN, BK, WARPS_M, WARPS_N, TM, TN, PAD_A>( \
            name, M, N, K, d_A, d_B, d_C, h_C_ref, h_C_test, warmup_iters, bench_iters, cublas_gflops))

    // 128x128 Block Tiles
    EVAL("128x128x16 | Warps 2x4 (WM=64, WN=32) | TM=8, TN=8 | Pad=4", 128, 128, 16, 2, 4, 8, 8, 4);
    EVAL("128x128x16 | Warps 4x2 (WM=32, WN=64) | TM=8, TN=8 | Pad=4", 128, 128, 16, 4, 2, 8, 8, 4);
    EVAL("128x128x16 | Warps 2x4 (WM=64, WN=32) | TM=8, TN=8 | Pad=0", 128, 128, 16, 2, 4, 8, 8, 0);
    EVAL("128x128x16 | Warps 4x2 (WM=32, WN=64) | TM=8, TN=8 | Pad=0", 128, 128, 16, 4, 2, 8, 8, 0);
    EVAL("128x128x16 | Warps 4x4 (WM=32, WN=32) | TM=8, TN=4 | Pad=4", 128, 128, 16, 4, 4, 8, 4, 4);
    EVAL("128x128x16 | Warps 4x4 (WM=32, WN=32) | TM=4, TN=8 | Pad=4", 128, 128, 16, 4, 4, 4, 8, 4);
    EVAL("128x128x8  | Warps 2x4 (WM=64, WN=32) | TM=8, TN=8 | Pad=4", 128, 128,  8, 2, 4, 8, 8, 4);
    EVAL("128x128x8  | Warps 4x2 (WM=32, WN=64) | TM=8, TN=8 | Pad=4", 128, 128,  8, 4, 2, 8, 8, 4);

    // 128x64 Block Tiles
    EVAL("128x64x16  | Warps 2x2 (WM=64, WN=32) | TM=8, TN=8 | Pad=4", 128,  64, 16, 2, 2, 8, 8, 4);
    EVAL("128x64x16  | Warps 4x2 (WM=32, WN=32) | TM=8, TN=4 | Pad=4", 128,  64, 16, 4, 2, 8, 4, 4);
    EVAL("128x64x16  | Warps 4x2 (WM=32, WN=32) | TM=4, TN=8 | Pad=4", 128,  64, 16, 4, 2, 4, 8, 4);

    // 64x128 Block Tiles
    EVAL("64x128x16  | Warps 2x2 (WM=32, WN=64) | TM=8, TN=8 | Pad=4",  64, 128, 16, 2, 2, 8, 8, 4);
    EVAL("64x128x16  | Warps 2x4 (WM=32, WN=32) | TM=4, TN=8 | Pad=4",  64, 128, 16, 2, 4, 4, 8, 4);
    EVAL("64x128x16  | Warps 2x4 (WM=32, WN=32) | TM=8, TN=4 | Pad=4",  64, 128, 16, 2, 4, 8, 4, 4);

    // 64x64 Block Tiles
    EVAL("64x64x16   | Warps 2x2 (WM=32, WN=32) | TM=4, TN=8 | Pad=4",  64,  64, 16, 2, 2, 4, 8, 4);
    EVAL("64x64x16   | Warps 2x2 (WM=32, WN=32) | TM=8, TN=4 | Pad=4",  64,  64, 16, 2, 2, 8, 4, 4);

    #undef EVAL

    // Sort by GFLOPS descending
    std::sort(results.begin(), results.end(), [](const ConfigResult& a, const ConfigResult& b) {
        return a.gflops > b.gflops;
    });

    // Print Leaderboard
    std::cout << std::left << std::setw(6)  << "Rank"
              << std::setw(60) << "Configuration"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(16) << "vs cuBLAS (%)"
              << std::setw(10) << "Status" << "\n";
    std::cout << std::string(119, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::string rank_str = (i == 0) ? " 1st" : (" #" + std::to_string(i + 1));
        std::cout << std::left << std::setw(6)  << rank_str
                  << std::setw(60) << r.name
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms
                  << std::setw(15) << std::setprecision(1) << r.gflops
                  << std::setw(16) << std::setprecision(1) << (std::to_string(static_cast<int>(r.pct_cublas)) + "%")
                  << std::setw(10) << (r.passed ? "PASSED" : "FAILED") << "\n";
    }
    std::cout << std::string(119, '-') << "\n\n";

    // Summary of Winner
    if (!results.empty()) {
        const auto& winner = results[0];
        std::cout << "🏆 [Auto-Tuner Golden Configuration Found]:\n";
        std::cout << "   - Configuration : " << winner.name << "\n";
        std::cout << "   - Block Tile    : BM=" << winner.BM << ", BN=" << winner.BN << ", BK=" << winner.BK << "\n";
        std::cout << "   - Warp Grid     : WARPS_M=" << winner.WARPS_M << ", WARPS_N=" << winner.WARPS_N << " (" << (winner.WARPS_M * winner.WARPS_N) << " warps)\n";
        std::cout << "   - Thread Tile   : TM=" << winner.TM << ", TN=" << winner.TN << "\n";
        std::cout << "   - Performance   : " << std::fixed << std::setprecision(3) << winner.time_ms << " ms | "
                  << std::setprecision(1) << winner.gflops << " GFLOPS (" << std::setprecision(1) << winner.pct_cublas << "% of cuBLAS)\n\n";
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_ref); free(h_C_test);
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int M = 2048, N = 2048, K = 2048;
    int warmup_iters = 5;
    int bench_iters = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "-n" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "-k" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "-w" && i + 1 < argc) warmup_iters = std::atoi(argv[++i]);
        else if (arg == "-r" && i + 1 < argc) bench_iters = std::atoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -m <int>          Matrix height M (default 2048)\n"
                      << "  -n <int>          Matrix width N (default 2048)\n"
                      << "  -k <int>          Matrix depth K (default 2048)\n"
                      << "  -w <int>          Warmup iterations (default 5)\n"
                      << "  -r <int>          Benchmark iterations (default 20)\n";
            return 0;
        }
    }

    run_autotune(M, N, K, warmup_iters, bench_iters);
    return 0;
}
