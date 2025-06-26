# HIP Runtime Benchmark Suite

This project is a lightweight HIP benchmarking suite designed to explore GPU runtime behavior on AMD hardware using the ROCm platform. For this project the Radeon W7900 PRO series GPU was used. 

### Goal of this Repo
The goal is to gain hands-on experience with HIP runtime APIs and measure key aspects of GPU execution such as:

- Kernel launch latency
- Memory transfer bandwidth (host-to-device, device-to-host, device-to-device)
- Kernel execution time for basic GPU workloads (vector addition, matrix multiplication, no-op)


This project was built as part of my effort to deepen my understanding of GPU runtimes.

---

## ✅ Benchmarked Kernels So Far:

| Kernel        | Description                           |
|---------------|---------------------------------------|
| Vector Add    | Simple element-wise vector addition   |
| No-op         | Kernel launch overhead measurement    |
| Matmul        | Basic dense matrix multiplication (NxN)|

---

## ✅ Features
- 📏 **Event-based GPU timing (using HIP events)**
- 🧪 **Result validation for correctness**
- ⚙️ **CMake-based build system**
- 🔌 **Supports configurable kernel launch sizes**
- 🧹 **Clean separation between kernels and host-side launch code**

---

## ✅ Build Instructions (Tested on ROCm 6.4.x)

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
