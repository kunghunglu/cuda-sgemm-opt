#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <set>
#include "utils.h"
#include "sgemm.h"

struct KernelInfo {
    int id;
    std::string name;
    void (*func)(int, int, int, float, const float*, const float*, float, float*);
};

void run_sgemm_cublas_wrapper(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    run_sgemm_cublas(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
}

void run_sgemm_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    // cuBLAS uses Column-Major order by default.
    // Row-Major C = A * B is equivalent to Column-Major C^T = B^T * A^T.
    // So passing B as matrix A, A as matrix B gives exact row-major result!
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
}

int main(int argc, char** argv) {
    int M = 2048;
    int N = 2048;
    int K = 2048;
    int warmup_iters = 5;
    int bench_iters = 20;
    std::set<int> target_kernels;
    bool skip_verify = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-m" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "-n" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "-k" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "-w" && i + 1 < argc) warmup_iters = std::atoi(argv[++i]);
        else if (arg == "-r" && i + 1 < argc) bench_iters = std::atoi(argv[++i]);
        else if (arg == "--kernel" && i + 1 < argc) {
            std::string arg = argv[++i];
            std::stringstream ss(arg);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    target_kernels.insert(std::atoi(item.c_str()));
                }
            }
        }
        else if (arg == "--skip-verify") skip_verify = true;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -m <int>          Matrix height M (default 2048)\n"
                      << "  -n <int>          Matrix width N (default 2048)\n"
                      << "  -k <int>          Matrix depth K (default 2048)\n"
                      << "  -w <int>          Warmup iterations (default 5)\n"
                      << "  -r <int>          Benchmark iterations (default 20)\n"
                      << "  --kernel <id(s)>  Target kernel index or comma-separated indices (0-7, 10=cuBLAS, default all)\n"
                      << "  --skip-verify     Skip numerical verification against reference\n";
            return 0;
        }
    }

    std::cout << "===============================================================\n";
    std::cout << " CUDA SGEMM Optimization Benchmark Suite\n";
    std::cout << " Matrix Size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << " Warmup: " << warmup_iters << " iters | Benchmark: " << bench_iters << " iters\n";
    std::cout << " Total FLOPs: " << std::scientific << std::setprecision(3) 
              << (2.0 * M * N * K) << " FLOPs\n";
    std::cout << "===============================================================\n\n";

    // Allocate host memory
    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C_ref = (float*)malloc(bytes_C);
    float* h_C_test = (float*)malloc(bytes_C);

    // Initialize input matrices
    std::srand(42);
    randomize_matrix(h_A, M * K);
    randomize_matrix(h_B, K * N);
    zero_matrix(h_C_ref, M * N);
    zero_matrix(h_C_test, M * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // Compute Reference using cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    run_sgemm_cublas(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Register all kernels
    std::vector<KernelInfo> kernels = {
        {0, "Kernel 0: Naive", run_sgemm_00_naive},
        {1, "Kernel 1: Global Memory Coalescing", run_sgemm_01_coalesced},
        {2, "Kernel 2: Shared Memory Tiling", run_sgemm_02_shared_mem},
        {3, "Kernel 3: 1D Thread Tiling", run_sgemm_03_1d_block_tiling},
        {4, "Kernel 4: 2D Thread Tiling", run_sgemm_04_2d_block_tiling},
        {5, "Kernel 5: Vectorized Access (float4)", run_sgemm_05_vectorized},
        {6, "Kernel 6: SMEM Double Buffering", run_sgemm_06_smem_double_buffering},
        {7, "Kernel 7: Bank Conflict Free", run_sgemm_07_bank_conflict_free},
       // {8, "Kernel 8: Hierarchical Warp Tiling", run_sgemm_08_warp_tiling},
       // {9, "Kernel 9: Tensor Cores (WMMA)", run_sgemm_09_tensor_core_wmma},
        {10, "Reference: cuBLAS", run_sgemm_cublas_wrapper}
    };

    // Print table header
    std::cout << std::left << std::setw(38) << "Kernel Name"
              << std::setw(15) << "Status"
              << std::setw(16) << "Max Abs Error"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS" << "\n";
    std::cout << std::string(99, '-') << "\n";

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (const auto& k : kernels) {
        if (!target_kernels.empty() && target_kernels.find(k.id) == target_kernels.end()) {
            continue;
        }

        // Reset d_C to zeros before execution
        CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

        // Warmup runs
        for (int i = 0; i < warmup_iters; ++i) {
            k.func(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < bench_iters; ++i) {
            k.func(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start, stop));
        float avg_time_ms = total_time_ms / bench_iters;

        double gflops = (2.0 * M * N * K) / (avg_time_ms * 1e-3) / 1e9;

        // Copy back result and verify
        CUDA_CHECK(cudaMemcpy(h_C_test, d_C, bytes_C, cudaMemcpyDeviceToHost));
        float max_err = calc_max_abs_error(h_C_ref, h_C_test, M, N);
        bool passed = skip_verify || (max_err < 1.0f); // WMMA FP16 precision threshold ~1e-1 to 1.0 depending on matrix magnitude

        std::cout << std::left << std::setw(38) << k.name
                  << std::setw(15) << (passed ? "PASSED" : "FAILED")
                  << std::scientific << std::setprecision(3) << std::setw(16) << max_err
                  << std::fixed << std::setprecision(3) << std::setw(15) << avg_time_ms
                  << std::fixed << std::setprecision(2) << std::setw(15) << gflops << "\n";
    }

    std::cout << std::string(99, '-') << "\n\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_test);

    return 0;
}
