#include <THC/THC.h>
#include "mathutil_cuda_kernel.h"

extern THCState *state;

int broadcast_sum(THCudaTensor *a_tensor, THCudaTensor *b_tensor, int x, int y)
{
    float *a = THCudaTensor_data(state, a_tensor);
    float *b = THCudaTensor_data(state, b_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

    broadcast_sum_cuda(a, b, x, y, stream);

    return 1;
}
