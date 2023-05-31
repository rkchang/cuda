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
}