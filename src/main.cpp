#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include "../kernels/vector_add.hpp"
#include "timer.hpp"

#define CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(e) << std::endl; \
        exit(1); \
    } \
} while (0)

//declare kernel
__global__ void vector_add(const float* A, const float* B, float* C, int N);

void validate(float* hC, int N);


int main() { 
    const int N = 1 << 20; 
    size_t size = N * sizeof(float);

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    hA = new float[N];
    hB = new float[N];
    hC = new float[N];
    
    for (int i = 0; i < N; i++){
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    CHECK(hipMalloc(&dA, size));
    CHECK(hipMalloc(&dB, size));
    CHECK(hipMalloc(&dC, size));
    CHECK(hipMemset(dC, 0, size));

    CHECK(hipMemcpy(dA, hA, size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB, size, hipMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x -1) / block.x);

    GpuTimer timer;
    timer.tic();
    hipLaunchKernelGGL(vector_add, grid, block, 0, 0, dA, dB, dC, N);
    CHECK(hipDeviceSynchronize());  
    float elapsed = timer.toc();

    CHECK(hipMemcpy(hC, dC, size, hipMemcpyDeviceToHost));
    validate(hC, N);
    std::cout << "Elapsed time: " << elapsed <<" ms " << std::endl;

    // free GPU memory
    CHECK(hipFree(dA));
    CHECK(hipFree(dB));
    CHECK(hipFree(dC));

    //free CPU memory
    delete[] hA;
    delete[] hB;
    delete[] hC;
    
    return 0;

}


void validate(float* hC, int N){
    bool valid = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(hC[i] - 3.0f) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": " << hC[i] << std::endl;
            valid = false;
            break;
        }
    }
    if (valid)
        std::cout << "Result valid ✅\n";
    else
        std::cout << "Result invalid ❌\n";
    
}