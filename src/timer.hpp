#pragma once
#include <hip/hip_runtime.h>

struct GpuTimer {
    hipEvent_t start_, stop_;

    GpuTimer() {
        hipEventCreate(&start_);
        hipEventCreate(&stop_);
    }

    ~GpuTimer() {
        hipEventDestroy(start_);
        hipEventDestroy(stop_);
    }

    void tic() {
        hipEventRecord(start_, 0);
    }

    float toc() {
        hipEventRecord(stop_, 0);
        hipEventSynchronize(stop_);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};
