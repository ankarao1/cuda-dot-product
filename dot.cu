#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

// ================= CUDA ERROR CHECK MACRO =================
#define CUDA_CHECK(call)                                      \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            std::cerr << "CUDA Error: "                        \
                      << cudaGetErrorString(err)               \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// ================= CUDA KERNEL =================
__global__
void dotProductKernel(double* A, double* B,
                      double* blockSums, size_t N)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    double localSum = 0.0;

    // Grid-stride loop
    while (idx < N) {
        localSum += A[idx] * B[idx];
        idx += stride;
    }

    // Store in shared memory
    sdata[tid] = localSum;
    __syncthreads();

    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block result
    if (tid == 0)
        blockSums[blockIdx.x] = sdata[0];
}

// ================= OPENMP VERSION =================
double dotProductOMP(double* A, double* B, size_t N)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; i++)
        sum += A[i] * B[i];

    return sum;
}

// ================= CUDA VERSION =================
double dotProductCUDA(double* A, double* B, size_t N,
                      double& totalTime,
                      double& kernelTime)
{
    double *d_A, *d_B, *d_blockSums;
    double result = 0.0;

    int blockSize = 256;
    int gridSize  = 1024;

    // ----------- TOTAL GPU TIME START -----------
    auto t0 = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_blockSums,
                          gridSize * sizeof(double)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, A,
                          N * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, B,
                          N * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ----------- KERNEL TIME START -----------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    dotProductKernel<<<gridSize, blockSize,
                       blockSize * sizeof(double)>>>(
        d_A, d_B, d_blockSums, N);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    kernelTime = ms / 1000.0;   // convert to seconds

    // ----------- COPY BACK BLOCK RESULTS -----------
    std::vector<double> h_blockSums(gridSize);

    CUDA_CHECK(cudaMemcpy(h_blockSums.data(),
                          d_blockSums,
                          gridSize * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    for (int i = 0; i < gridSize; i++)
        result += h_blockSums[i];

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_blockSums));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    auto t1 = std::chrono::high_resolution_clock::now();
    totalTime =
        std::chrono::duration<double>(t1 - t0).count();

    return result;
}

// ================= MAIN =================
int main()
{
    const size_t N = 100000000;  // 100 Million

    std::cout << "Vector size: " << N << std::endl;

    std::vector<double> A(N), B(N);

    // Random initialization
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < N; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    // ================= OPENMP =================
    auto t0 = std::chrono::high_resolution_clock::now();
    double ompResult = dotProductOMP(A.data(),
                                     B.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ompTime =
        std::chrono::duration<double>(t1 - t0).count();

    // ================= CUDA =================
    double totalTimeCUDA = 0.0;
    double kernelTimeCUDA = 0.0;

    double cudaResult =
        dotProductCUDA(A.data(), B.data(), N,
                       totalTimeCUDA,
                       kernelTimeCUDA);

    // ================= OUTPUT =================
    std::cout << "\nResults:\n";
    std::cout << "OMP Result:   " << ompResult << "\n";
    std::cout << "CUDA Result:  " << cudaResult << "\n\n";

    std::cout << "Timings:\n";
    std::cout << "OpenMP Time:        "
              << ompTime << " s\n";

    std::cout << "CUDA Total Time:    "
              << totalTimeCUDA << " s\n";

    std::cout << "CUDA Kernel Time:   "
              << kernelTimeCUDA << " s\n\n";

    std::cout << "Speedup (Total):    "
              << ompTime / totalTimeCUDA << "\n";

    std::cout << "Speedup (Kernel):   "
              << ompTime / kernelTimeCUDA << "\n";

    return 0;
}