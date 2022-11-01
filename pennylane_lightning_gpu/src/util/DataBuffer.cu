#pragma once
#include "DataBuffer.hpp"
#include <cuComplex.h>
#include <cuda.h>

namespace Pennylane::CUDA {

template <class DeviceDataT, class index_type>
__global__ void cuda_element_set(index_type num_indices, DeviceDataT *value,
                                 index_type *indices,
                                 DeviceDataT *gpu_buffer_) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        gpu_buffer_[indices[i]] = value[i];
    }
}

// template <class GPUDataT,class DevTagT = int>
template <class GPUDataT, class DevTagT>
template <class DeviceDataT, class index_type>
void DataBuffer<GPUDataT, DevTagT>::setElements(index_type &num_indices,
                                                DeviceDataT *value,
                                                index_type *indices,
                                                bool async) {
    const size_t thread_per_block = 1024;
    const size_t num_blocks = num_indices / thread_per_block + 1;
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(num_blocks, 1);

    cuda_element_set<DeviceDataT, index_type>
        <<<gridSize, blockSize>>>(num_indices, value, indices, gpu_buffer_);
}
} // namespace Pennylane::CUDA
