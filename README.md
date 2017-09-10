# Pytorch Custom CUDA kernel Tutorial

This repository contains a tutorial code for making a custom CUDA function for
pytorch. The code is based on the pytorch [C extension
example](https://github.com/pytorch/extension-ffi).

**Disclaimer**

`This tutorial was written when pytorch did not support broadcasting sum. Now that it supports, probably you wouldn't need to make your own broadcasting sum function, but you can still follow the tutorial to build your own custom layer with a custom CUDA kernel.`

In this repository, we will build a simple CUDA based broadcasting sum
function.  The current version of pytorch does not support [broadcasting
sum](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html), thus we
have to manually expand a tensor like using `expand_as` which makes a new
tensor and takes additional memory and computation.

For example,

```python
a = torch.randn(3, 5)
b = torch.randn(3, 1)
# The following line will give an error
# a += b

# Expand b to have the same dimension as a
b_like_a = b.expand_as(a)
a += b_like_a
```

In this post, we will build a function that can compute `a += b` without
explicitly expanding `b`.

```python
mathutil.broadcast_sum(a, b, *map(int, a.size()))
```

## Make a CUDA kernel

First, let's make a cuda kernel that adds `b` to `a` without making a copy of a tensor `b`.

```cuda
__global__ void broadcast_sum_kernel(float *a, float *b, int x, int y, int size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= size) return;
    int j = i % x; i = i / x;
    int k = i % y;
    a[IDX2D(j, k, y)] += b[k];
}
```

## Make a C wrapper

Once you made a CUDA kernel, you have to wrap it with a C code. However, we are not using the pytorch backend yet. Note that the inputs are already device pointers.


```c++
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
```

## Connect Pytorch backends with the C Wrapper

Next, we have to connect the pytorch backend with our C wrapper. You can expose the device pointer using the function `THCudaTensor_data`. The pointers `a` and `b` are device pointers (on GPU).


```c++
extern THCState *state;

int broadcast_sum(THCudaTensor *a_tensor, THCudaTensor *b_tensor, int x, int y)
{
    float *a = THCudaTensor_data(state, a_tensor);
    float *b = THCudaTensor_data(state, b_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

    broadcast_sum_cuda(a, b, x, y, stream);

    return 1;
}
```

## Make a python wrapper

Now that we built the cuda function and a pytorch function, we need to expose the function to python so that we can use the function in python.

We will first build a shared library using `nvcc`.

```shell
nvcc ... -o build/mathutil_cuda_kernel.so src/mathutil_cuda_kernel.cu
```

Then, we will use the pytorch `torch.utils.ffi.create_extension` function which automatically put appropriate headers and builds a python loadable shared library.

```python
from torch.utils.ffi import create_extension

...

ffi = create_extension(
    'mathutils',
    headers=[...],
    sources=[...],
    ...
)

ffi.build()
```


## Test!

Finally, we can test our function by building it.
In the readme, I removed a lot of details, but you can see a working example.

```shell
git clone https://github.com/chrischoy/pytorch-cffi-tutorial
cd pytorch-cffi-tutorial
make
```

## Note

The function only takes `THCudaTensor`, which is `torch.FloatTensor().cuda()` in python.
