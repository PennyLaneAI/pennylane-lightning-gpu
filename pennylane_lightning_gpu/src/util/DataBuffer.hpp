#include "cuda.h"
#include "cuda_helpers.hpp"

namespace Pennylane::CUDA {

/**
 * @brief Data storage class for CUDA memory. Maintsins an associated stream and
 * device ID taken during time of allocation.
 *
 * @tparam DataT Data type to store.
 */
template <class GPUDataT> class DataBuffer {
  public:
    /**
     * @brief Construct a new DataBuffer object
     *
     * @param length Number of elements in data buffer.
     * @param device_id Associated device ID. Must be `cudaSetDevice`
     * compatible.
     * @param stream_id Associated stread ID. Must be `cudaSetDevice`
     * compatible.
     * @param alloc_memory Indicate whether to allocate the memory for the
     * buffer. Defaults to `true`
     */
    DataBuffer(std::size_t length, int device_id = 0,
               cudaStream_t stream_id = 0, bool alloc_memory = true)
        : length_{length}, device_id_{device_id}, stream_id_{stream_id} {
        if (alloc_memory && length > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_),
                           sizeof(GPUDataT) * length));
        }
    }

    // Move CTOR should be forbidden for CUDA memory; explicit copies only
    DataBuffer(DataBuffer &&other) = delete;
    DataBuffer(const DataBuffer &other) = delete;

    virtual ~DataBuffer() { PL_CUDA_IS_SUCCESS(cudaFree(gpu_buffer_)); };

    auto getData() -> GPUDataT * { return gpu_buffer_; }
    auto getData() const -> const GPUDataT * { return gpu_buffer_; }
    auto getLength() const { return length_; }

    /**
     * @brief Get the CUDA stream for the given object.
     *
     * @return const cudaStream_t&
     */
    inline auto getStream() const -> const cudaStream_t & { return stream_id_; }
    void setStream(const cudaStream_t &s) { stream_id_ = s; }

    /**
     * @brief Copy data from another GPU memory block to here.
     *
     */
    void CopyGpuDataToGpu(const GPUDataT *gpu_in, std::size_t length,
                          bool async = false) {
        PL_ABORT_IF_NOT(
            getLength() < length,
            "Sizes do not match for GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (async) {
            PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                getData(), gpu_in, sizeof(GPUDataT) * getLength(),
                cudaMemcpyDeviceToDevice, getStream()));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(getData(), gpu_in,
                                          sizeof(GPUDataT) * getLength(),
                                          cudaMemcpyDefault));
        }
    }

    /**
     * @brief Copy data from another GPU memory block to here.
     *
     */
    void CopyGpuDataToGpu(const DataBuffer &buffer, bool async = false) {
        CopyGpuDataToGpu(buffer.getData(), buffer.getLength(), async);
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     */
    template <class HostDataT = GPUDataT>
    void CopyHostDataToGpu(const HostDataT *host_in, std::size_t length,
                           bool async = false) {
        PL_ABORT_IF_NOT(
            (getLength() * sizeof(GPUDataT)) < (length * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (async) {
            PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                getData(), host_in, sizeof(GPUDataT) * getLength(),
                cudaMemcpyHostToDevice, getStream()));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(getData(), host_in,
                                          sizeof(GPUDataT) * getLength(),
                                          cudaMemcpyDefault));
        }
    }

    /**
     * @brief Explicitly copy data from GPU device to host memory.
     *
     */
    template <class HostDataT = GPUDataT>
    inline void CopyGpuDataToHost(HostDataT *host_out, std::size_t length,
                                  bool async = false) const {
        PL_ABORT_IF_NOT(
            (getLength() * sizeof(GPUDataT)) > (length * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (!async) {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(host_out, getData(),
                                          sizeof(GPUDataT) * getLength(),
                                          cudaMemcpyDefault));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                host_out, getData(), sizeof(GPUDataT) * getLength(),
                cudaMemcpyDeviceToHost, getStream()));
        }
    }

  private:
    std::size_t length_;
    int device_id_;
    cudaStream_t stream_id_;
    GPUDataT *gpu_buffer_;
};

} // namespace Pennylane::CUDA