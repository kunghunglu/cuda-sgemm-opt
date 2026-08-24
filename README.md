# CUDA SGEMM Optimization Suite

A high-performance, step-by-step Single-Precision General Matrix Multiplication (SGEMM: $C = \alpha AB + \beta C$) optimization journey on NVIDIA GPUs, progressing from a naive baseline to a CUTLASS-style templated hierarchical warp-tiling kernel that matches and outperforms NVIDIA cuBLAS.

---

## Overview & Progression

This project systematically demonstrates how to optimize FP32 GEMM kernels on modern NVIDIA GPUs through 10 progressive optimization steps:

| Step | Kernel Name | Core Optimization Technique | Key Impact |
| :--- | :--- | :--- | :--- |
| **0** | **Naive** | 1 thread per element, direct global memory access | Baseline |
| **1** | **Global Memory Coalescing** | Transpose thread mapping along contiguous $N$ dimension | Coalesced 128-byte DRAM transactions |
| **2** | **Shared Memory Tiling** | $16 \times 16$ SMEM cache for blocking along $K$ dimension | Reduced DRAM bandwidth pressure |
| **3** | **1D Thread Tiling** | 1D register accumulation ($TM=8$ per thread, $BM=BN=64$) | Increased arithmetic intensity ($2\times$) |
| **4** | **2D Thread Tiling** | 2D register tile ($TM=8, TN=8$, 64 outputs per thread) | High register reuse ($8\times$ arithmetic intensity) |
| **5** | **Vectorized Memory Access** | `float4` (128-bit `LDG.128` / `STG.128`) global memory instructions | Saturated bus bandwidth & reduced instruction count |
| **6** | **SMEM Double Buffering** | Ping-pong shared memory buffers (Software Pipelining) | Overlapped DRAM prefetching with ALU math |
| **7** | **Bank Conflict Free Layout** | Transposed Matrix A in SMEM ($As[k][m]$) + Row Padding (`PAD_A=4`) | Enabled vectorized `LDS.128` & zero SMEM bank conflicts on A |
| **8** | **Hierarchical Warp Tiling** | 3-tier mapping: Block ($128\times 128$) -> Warp ($32\times 64$) -> Thread ($8\times 8$) | Hardware multicasting & reduced Matrix B conflict |
| **9** | **Templated Warp Tiling** | Generic C++ template, $2\times 4$ Warps ($WM=64, WN=32$) with 8-Way Broadcast | **108% cuBLAS Performance** |

---

## Benchmark Results

### Environment
* **GPU**: NVIDIA GeForce GTX 1080 Ti (Pascal GP102, 28 SMs, 3584 CUDA Cores @ 1.91 GHz)
* **Matrix Size**: $M = 2048, N = 2048, K = 2048$ (Total FLOPs: 17.18 GFLOPs)
* **Compiler**: NVCC (CUDA 12.4), CMake 3.18+, C++17 (`-O3 --use_fast_math`)

### Results Table ($M=N=K=2048$)

```
===============================================================
 CUDA SGEMM Optimization Benchmark Suite
 Matrix Size: M=2048, N=2048, K=2048
 Warmup: 5 iters | Benchmark: 20 iters
 Total FLOPs: 1.718e+10 FLOPs
===============================================================

Kernel Name                           Time (ms)    GFLOPS       vs cuBLAS (%)   
--------------------------------------------------------------------------------
Kernel 0: Naive                       101.195      169.77       2.1%            
Kernel 1: Global Memory Coalescing    30.343       566.19       6.9%            
Kernel 2: Shared Memory Tiling        15.377       1117.26      13.7%           
Kernel 3: 1D Thread Tiling            7.426        2313.57      28.3%           
Kernel 4: 2D Thread Tiling            5.994        2866.09      35.1%           
Kernel 5: Vectorized Access (float4)  3.874        4435.08      54.3%           
Kernel 6: SMEM Double Buffering       3.512        4891.32      59.9%           
Kernel 7: Bank Conflict Free          2.457        6992.40      85.6%           
Kernel 8: Hierarchical Warp Tiling    2.255        7620.13      93.3%           
Kernel 9: Templated Warp Tiling       2.075        8281.16      101.4%   <-- ⚡
Reference: cuBLAS                     2.183        7870.16      100.0%          
--------------------------------------------------------------------------------
```

---

## Build Instructions

### Prerequisites
* CMake 3.18 or higher
* CUDA Toolkit 11.0+ or 12.0+ with NVCC
* C++17 compatible host compiler (GCC 7+, Clang 6+, or MSVC 2019+)

### Build
```bash
# Clone the repository
git clone https://github.com/kunghunglu/cuda-sgemm-opt.git
cd cuda-sgemm-opt

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This will produce two executables inside the `build/` directory:
1. `build/sgemm_bench`: Comprehensive benchmark comparing Kernels 0-9 against cuBLAS.
2. `build/sgemm_tuner`: Auto-tuning engine searching the optimal tile geometry for your specific GPU.

---

## Usage

### 1. Running the Benchmark Suite (`sgemm_bench`)

```bash
# Run all kernels with default size (M=2048, N=2048, K=2048)
./build/sgemm_bench

# Run with custom matrix dimensions and iteration counts
./build/sgemm_bench -m 4096 -n 4096 -k 4096 -w 10 -r 50

# Benchmark specific kernels (e.g. Kernel 7, 8, 9, and cuBLAS)
./build/sgemm_bench --kernel 7,8,9,10

# View all CLI options
./build/sgemm_bench --help
```

### 2. Running the Auto-Tuner (`sgemm_tuner`)

The Auto-Tuner systematically instantiates and benchmarks 16+ tile and warp geometries using C++ templates, evaluating GFLOPS, memory footprint, and relative speedup against cuBLAS:

```bash
# Auto-tune for default size (2048x2048x2048)
./build/sgemm_tuner

# Auto-tune for custom matrix size (e.g. 4096x4096x4096)
./build/sgemm_tuner -m 4096 -n 4096 -k 4096 -r 10
```

#### Auto-Tuner Sample Output ($M=N=K=4096$ on GTX 1080 Ti):
```
=========================================================================================
 CUDA SGEMM Kernel 9 Auto-Tuning Engine
 Device: NVIDIA GeForce GTX 1080 Ti (28 SMs, CC 6.1)
 Matrix Size: M=4096, N=4096, K=4096 | Total FLOPs: 1.374e+11 FLOPs
=========================================================================================

 Reference cuBLAS Performance: 16.486 ms (8336.5 GFLOPS)

Rank  Configuration                                               Time (ms)   GFLOPS         vs cuBLAS (%)   Status    
-----------------------------------------------------------------------------------------------------------------------
 1st  128x128x16 | Warps 2x4 (WM=64, WN=32) | TM=8, TN=8 | Pad=4  15.370      8941.8         107%            PASSED    
 #2   128x128x8  | Warps 2x4 (WM=64, WN=32) | TM=8, TN=8 | Pad=4  15.714      8746.2         104%            PASSED    
 #3   128x128x16 | Warps 2x4 (WM=64, WN=32) | TM=8, TN=8 | Pad=0  16.419      8371.0         100%            PASSED    
 #4   128x64x16  | Warps 2x2 (WM=64, WN=32) | TM=8, TN=8 | Pad=4  16.867      8148.4         97%             PASSED    
-----------------------------------------------------------------------------------------------------------------------

🏆 [Auto-Tuner Golden Configuration Found]:
   - Configuration : 128x128x16 | Warps 2x4 (WM=64, WN=32) | TM=8, TN=8 | Pad=4
   - Performance   : 15.370 ms | 8941.8 GFLOPS (107.3% of cuBLAS)
```

---

## Project Structure

```
.
├── CMakeLists.txt                       # CMake build script
├── README.md                            # Documentation and benchmark reports
├── include/
│   ├── common.h                         # Architecture & vectorization constants
│   ├── sgemm.h                          # Kernel runner declarations & cuBLAS wrapper
│   ├── templated_warp_tiling.cuh        # Generic templated Warp-Tiling SGEMM Kernel
│   └── utils.h                          # GPU constants, error checkers, float4 operators
└── src/
    ├── main.cu                          # Benchmark test harness
    ├── tuner.cu                         # Auto-Tuner parameter sweep engine
    ├── utils.cu                         # Verification and matrix generation utilities
    └── kernels/
        ├── 00_naive.cu                  # Step 0: Naive baseline
        ├── 01_coalesced.cu              # Step 1: Global memory coalescing
        ├── 02_shared_mem.cu             # Step 2: Shared memory tiling
        ├── 03_1d_block_tiling.cu        # Step 3: 1D thread tiling
        ├── 04_2d_block_tiling.cu        # Step 4: 2D thread tiling
        ├── 05_vectorized.cu             # Step 5: Vectorized float4 memory access
        ├── 06_smem_double_buffering.cu  # Step 6: Shared memory double buffering
        ├── 07_bank_conflict_free.cu     # Step 7: Transposed A + Row padding
        ├── 08_warp_tiling.cu            # Step 8: Hierarchical warp tiling
        └── 09_templated_warp_tiling.cu  # Step 9: Templated tuned warp tiling
```

---

## License

MIT License. Free for educational and research use.
