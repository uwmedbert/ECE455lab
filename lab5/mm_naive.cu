// mm_naive.cu
// Naive CUDA matrix multiplication kernel: each thread computes one element C[i, j]

#include <cassert>
#include <iostream>

// Naive kernel: each thread computes one element C[i, j]
template <typename T>
__global__ void mm_kernel(const T* mat_1, const T* mat_2, T* mat_3,
                          size_t m, size_t n, size_t p) {
    // Compute (i, j) coordinates from 2D grid
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (i >= m || j >= p) return;

    // Compute dot product of row i (A) and column j (B)
    T acc_sum = 0;
    for (size_t k = 0; k < n; ++k)
        acc_sum += mat_1[i * n + k] * mat_2[k * p + j];

    // Write result
    mat_3[i * p + j] = acc_sum;
}

// --- Host driver: run tests and measure kernel performance ---
int main() {
    const size_t num_tests = 2; // Correctness trials

    // Correctness checks (assumed to be defined elsewhere)
    assert(random_multiple_test_mm_cuda<int32_t>(num_tests));
    assert(random_multiple_test_mm_cuda<float>(num_tests));
    assert(random_multiple_test_mm_cuda<double>(num_tests));

    std::cout << "All tests passed!\n";

    // --- Performance measurement ---
    const size_t num_measurement_tests = 2;
    const size_t num_measurement_warmups = 1;

    size_t m = MAT_DIM, n = MAT_DIM, p = MAT_DIM;

    // Measure average latency across data types
    float mm_cuda_int32_latency = measure_latency_mm_cuda<int32_t>(
        m, n, p, num_measurement_tests, num_measurement_warmups);

    float mm_cuda_float_latency = measure_latency_mm_cuda<float>(
        m, n, p, num_measurement_tests, num_measurement_warmups);

    float mm_cuda_double_latency = measure_latency_mm_cuda<double>(
        m, n, p, num_measurement_tests, num_measurement_warmups);

    // Print results
    std::cout << "Matrix Multiplication Runtime\n";
    std::cout << "m: " << m << " n: " << n << " p: " << p << "\n";
    std::cout << "INT32 : " << mm_cuda_int32_latency << " ms\n";
    std::cout << "FLOAT : " << mm_cuda_float_latency << " ms\n";
    std::cout << "DOUBLE: " << mm_cuda_double_latency << " ms\n";

    return 0;
}
