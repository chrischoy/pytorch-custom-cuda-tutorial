#ifndef _MATHUTIL_CUDA_KERNEL
#define _MATHUTIL_CUDA_KERNEL

#define IDX2D(i, j, dj) (dj * i + j)
#define IDX3D(i, j, k, dj, dk) (IDX2D(IDX2D(i, j, dj), k, dk))

#define BLOCK 512
#define MAX_STREAMS 512

#ifdef __cplusplus
extern "C" {
#endif

void broadcast_sum_cuda(float *a, float *b, int x, int y, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
