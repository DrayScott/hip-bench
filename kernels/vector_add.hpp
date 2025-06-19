#pragma once
__global__ void vector_add(const float* A,
                           const float* B,
                           float*       C,
                           int          N);
