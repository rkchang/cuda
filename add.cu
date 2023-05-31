#include <iostream>
#include <math.h>

// __global functions are kernels
// They run on the GPU
// kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void) {
  int N = 1 << 20; // 1M elements

  // Allocate Unified Memory – accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // CUDA GPUS have parallel processors grouped into streaming multiprocessors
  // (SMs) each SM can run multiple concurrent thread blocks
  // Run kernel on 1M elements on the GPU
  // <<<x,y>>> means kernel launch
  // x = number of thread blocks
  // y = number of parallel threads in a thread block to use for the launch on
  // the GPU
  // - each block is a multiple of 32 in size
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  std::cout << "N: " << N << std::endl;
  std::cout << "divdend: " << (N + blockSize - 1) << std::endl;
  std::cout << "numBlocks: " << numBlocks << std::endl;
  add<<<numBlocks, 256>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}