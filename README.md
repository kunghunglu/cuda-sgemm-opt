# CUDA SGEMM Optimization Suite


## Benchmark Results

### Environment
* **GPU**: NVIDIA GeForce GTX 1080 Ti (Pascal GP102, 28 SMs, 3584 CUDA Cores @ 1.91 GHz)
* **Matrix Size**: $M = 2048, N = 2048, K = 2048$ (Total FLOPs: 17.18 GFLOPs)
* **Compiler**: NVCC (CUDA 12.4), CMake 3.18+, C++17 (`-O3 --use_fast_math`)

### Performance Summary ($M=N=K=2048$)

```
===============================================================
 CUDA SGEMM Optimization Benchmark Suite
 Matrix Size: M=2048, N=2048, K=2048
 Warmup: 5 iters | Benchmark: 20 iters
 Total FLOPs: 1.718e+10 FLOPs
===============================================================

Kernel Name                           Time (ms)    GFLOPS       vs cuBLAS (%)   
--------------------------------------------------------------------------------
Kernel 0: Naive                       105.341      163.09       1.6%            
Kernel 1: Global Memory Coalescing    31.454       546.19       5.3%            
Kernel 2: Shared Memory Tiling        15.706       1093.83      10.7%           
Kernel 3: 1D Thread Tiling            7.501        2290.47      22.4%           
Kernel 4: 2D Thread Tiling            5.631        3050.79      29.9%           
Kernel 5: Vectorized Access (float4)  3.997        4297.77      42.1%           
Kernel 6: SMEM Double Buffering       3.708        4633.18      45.3%           
Kernel 7: Bank Conflict Free          2.459        6986.14      68.4%           
Kernel 8: Hierarchical Warp Tiling    2.570        6684.42      65.4%           
Kernel 9: Templated Warp Tiling       1.825        9412.71      92.1%   <-- ⚡
Reference: cuBLAS                     1.681        10219.78     100.0%          
--------------------------------------------------------------------------------
```

---

## Overview

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
| **9** | **Templated Warp Tiling** | Generic C++ template, $2\times 4$ Warps ($WM=64, WN=32$) with 8-Way Broadcast | **92% cuBLAS Performance** |

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

### Run
```bash
# Run full benchmark suite
./build/sgemm_bench

# Run Auto-Tuner parameter sweep
./build/sgemm_tuner
```
