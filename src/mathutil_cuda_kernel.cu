#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "mathutil_cuda_kernel.h"

dim3 cuda_gridsize(int n)
{
    int k = (n - 1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    return d;
}

__global__ void broadcast_sum_kernel(float *a, float *b, int x, int y, int size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= size) return;
    int j = i % y; i = i / y;
    int k = i % x;
    a[IDX2D(k, j, y)] += b[k];
}

void broadcast_sum_cuda(float *a, float *b, int x, int y, cudaStream_t stream)
{
    int size = x * y;
    cudaError_t err;

    broadcast_sum_kernel<<<cuda_gridsize(size), BLOCK, 0, stream>>>(a, b, x, y, size);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
