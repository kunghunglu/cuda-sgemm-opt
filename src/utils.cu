#include "utils.h"

void randomize_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
}

void zero_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = 0.0f;
    }
}

void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

float calc_max_abs_error(const float* refC, const float* testC, int M, int N) {
    float max_err = 0.0f;
    int size = M * N;
    for (int i = 0; i < size; ++i) {
        float err = std::abs(refC[i] - testC[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

bool verify_matrix(const float* refC, const float* testC, int M, int N, float tolerance) {
    int size = M * N;
    int error_count = 0;
    float max_err = 0.0f;

    for (int i = 0; i < size; ++i) {
        float diff = std::abs(refC[i] - testC[i]);
        float rel_err = diff / (std::abs(refC[i]) + 1e-5f);
        if (diff > tolerance && rel_err > tolerance) {
            if (error_count < 10) {
                std::printf("Mismatch at index %d (row %d, col %d): Ref = %f, Test = %f, Diff = %f\n",
                            i, i / N, i % N, refC[i], testC[i], diff);
            }
            error_count++;
        }
        if (diff > max_err) max_err = diff;
    }

    if (error_count > 0) {
        std::printf("Total mismatching elements: %d / %d (Max Abs Error: %e)\n", error_count, size, max_err);
        return false;
    }
    return true;
}
