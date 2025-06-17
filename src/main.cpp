#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>


#define CHECK(cmd) {
    hipError_t e = cmd; 
    if (e != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(e) << std::endl; exit(1); } }


int main() {
    const int N = 1 << 20; 
    size_t size N * sizeof(float);

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    hA = new float[N];
    hB = new float[N];
    hC = new float[N];
    
    for (int i = 0; i < N; i++){
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    CHECK(hipMalloc(&dA, size))
    CHECK(hipMalloc(&dB, size))
    CHECK(hipMalloc(&dC, size))

    CHECK(hipMemcpy(dA, hA, size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB, size, hipMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x -1) / block.x);

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(vector_add, grid, block, 0, 0, dA, dB, dC, N);
    CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    CHECK(hipMemcpy(hC, dC, size, hipMemcpyDeviceToHost));

    std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    CHECK(hipFree(dA));
    CHECK(hipFree(dB));
    CHECK(hipFree(dC));

    delete[] hA;
    delete[] hB;
    delete[] hC;
    return 0;

}


