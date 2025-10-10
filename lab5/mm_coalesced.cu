// mm_coalesced.cu
// Coalesced kernel: swapped x/y mapping improves memory access

#include <cassert>
#include <iostream>

// Coalesced kernel: each thread computes one element C[i, j]
// Mapping is swapped to improve global memory coalescing
template <typename T>
__global__ void mm_coalesced_kernel(const T* mat_1, const T* mat_2, T* mat_3,
                                    size_t m, size_t n, size_t p) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x; // columns -> x
    size_t i = blockIdx.y * blockDim.y + threadIdx.y; // rows -> y

    if (i >= m || j >= p) return;

    // Threads now traverse rows/cols in contiguous order
    T acc_sum = 0;
    for (size_t k = 0; k < n; ++k)
        acc_sum += mat_1[i * n + k] * mat_2[k * p + j];

    mat_3[i * p + j] = acc_sum;
}

// --- Host driver: same structure as mm_naive.cu ---
int main() {
    const size_t num_tests = 2;

    // Correctness tests (functions defined elsewhere)
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
