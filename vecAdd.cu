#include <iostream>

__global__ void vecAddKernel(float *C, float *A, float *B, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  int size = 1 << 20;
  float *C, *A, *B;
  cudaMallocManaged(&C, sizeof(float) * size);
  cudaMallocManaged(&A, sizeof(float) * size);
  cudaMallocManaged(&B, sizeof(float) * size);

  for (int i = 0; i < size; i++) {
    A[i] = 1;
    B[i] = 2;
  }

  int numBlocks = ceil(size / 256.0);
  int numThreads = 256;
  vecAddKernel<<<numBlocks, numThreads>>>(C, A, B, size);
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < size; i++)
    maxError = fmax(maxError, fabs(C[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(C);
  cudaFree(A);
  cudaFree(B);
}