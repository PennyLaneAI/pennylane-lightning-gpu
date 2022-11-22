#include "DataBuffer.hpp"

namespace Pennylane::CUDA {
// Explicit instantiation
template void DataBuffer<cuComplex, int>::setElements(int &num_indices,
                                                      cuComplex *value,
                                                      int *indices,
                                                      size_t thread_per_block,
                                                      cudaStream_t stream_id);
template void DataBuffer<double2, int>::setElements(long &num_indices,
                                                    cuDoubleComplex *value,
                                                    long *indices,
                                                    size_t thread_per_block,
                                                    cudaStream_t stream_id);
template <class DeviceDataT, class index_type>
__global__ void cuda_element_set(index_type num_indices, DeviceDataT *value,
                                 index_type *indices,
                                 DeviceDataT *gpu_buffer_) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        gpu_buffer_[indices[i]] = value[i];
    }
}

template <class GPUDataT, class DevTagT>
template <class DeviceDataT, class index_type>
void DataBuffer<GPUDataT, DevTagT>::setElements(index_type &num_indices,
                                                DeviceDataT *value,
                                                index_type *indices,
                                                size_t thread_per_block,
                                                cudaStream_t stream_id) {
    const size_t num_blocks = num_indices / thread_per_block + 1;
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(num_blocks, 1);

    cuda_element_set<DeviceDataT, index_type>
        <<<gridSize, blockSize, 0, stream_id>>>(num_indices, value, indices,
                                                gpu_buffer_);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}
} // namespace Pennylane::CUDA
