#include "cuda_helper.hpp"
#include <cuComplex.h>

namespace {
template <class DeviceDataT, class index_type>
__global__ void setStateVectorkernel(index_type num_indices, DeviceDataT *value,
                                     index_type *indices,
                                     DeviceDataT *gpu_buffer_) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        gpu_buffer_[indices[i]] = value[i];
    }
}

template <class GPUDataT, class index_type>
void setStateVector_CUDA(GPUDataT *sv, index_type &num_indices, GPUDataT *value,
                         index_type *indices, size_t thread_per_block,
                         cudaStream_t stream_id) {
    auto dv = std::div(num_indices, thread_per_block);
    const size_t num_blocks = dv.quot + (dv.rem == 0 ? 0 : 1);
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(num_blocks, 1);

    setStateVectorkernel<DeviceDataT, index_type>
        <<<gridSize, blockSize, 0, stream_id>>>(sv, num_indices, value, indices,
                                                gpu_buffer_);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}

template <class GPUDataT>
void setBasisState_CUDA(GPUDataT *sv, GPUDataT &value, size_t &index,
                        bool async, cudaStream_t stream_id) {
    if (!async) {
        PL_CUDA_IS_SUCCESS(cudaMemcpy(&sv[index], &value, sizeof(GPUDataT),
                                      cudaMemcpyHostToDevice));
    } else {
        PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(&sv[index], &value, sizeof(GPUDataT),
                                           cudaMemcpyHostToDevice, stream_id));
    }
}

} // namespace

namespace Pennylane {
// Explicit instantiation
template void setStateVector_CUDA(cuComplex *sv, int &num_indices,
                                  cuComplex *value, int *indices,
                                  size_t thread_per_block,
                                  cudaStream_t stream_id);
template void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                                  cuDoubleComplex *value, long *indices,
                                  size_t thread_per_block,
                                  cudaStream_t stream_id);

template void setBasisState_CUDA(cuComplex *sv, cuComplex &value, size_t &index,
                                 bool async, cudaStream_t stream_id);
template void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                                 size_t &index, bool async,
                                 cudaStream_t stream_id);

} // namespace Pennylane