#include "cuda_helpers.hpp"
#include <cuComplex.h>

namespace {

template <typename T>
__global__ void init_sv_kernel(T* x, int64_t n)
{
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i == 0)
    {
        x[i].x = 1.0;
        x[i].y = 0.0;
    }
    else if ( i < n )
    {
        x[i].x = 0.0;
        x[i].y = 0.0;
    }
}

template <typename T>
void call_init_kernel(T* x, int64_t n, cudaStream_t stream)
{
    constexpr int64_t threads = 256;
    const int64_t blocks  = n / threads + (n % threads == 0 ? 0 : 1);
    init_sv_kernel<<<blocks, threads, 0, stream>>>(x, n);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}
}

namespace Pennylane {
void initialize_sv(cuComplex* x, int64_t n, cudaStream_t stream)
{
    call_init_kernel(x,n,stream);
}

void initialize_sv(cuDoubleComplex* x, int64_t n, cudaStream_t stream)
{
    call_init_kernel(x,n,stream);
}
}
