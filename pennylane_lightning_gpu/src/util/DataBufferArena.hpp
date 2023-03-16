#pragma once

#include "DevTag.hpp"
#include "cuda.h"
#include "cuda_helpers.hpp"

namespace Pennylane::CUDA {

template <class GPUDataT, class DevTagT = int> class DataBufferArena {
  public:
    using type = GPUDataT;

    DataBufferArena(const std::vector<std::size_t> &buffer_lengths,
                    int device_id = 0, cudaStream_t stream_id = 0,
                    bool alloc_memory = true)
        : lengths_{buffer_lengths},
          gpu_buffer_offsets_(buffer_lengths.size() + 1, 0),
          dev_tag_{device_id, stream_id}, gpu_buffer_begin_{nullptr},
          gpu_buffer_end_{nullptr} {
        // Define the pointer offsets for each buffer
        std::partial_sum(buffer_lengths.cbegin(), buffer_lengths.cend(),
                         gpu_buffer_offsets_.begin() + 1);
        total_length_ = gpu_buffer_offsets_.back();
        if (alloc_memory && (buffer_lengths.size() > 0) &&
            (buffer_lengths[0] > 0)) {
            // Ensure we tag the current GPU
            dev_tag_.refresh();
            // Allocate the total buffer size worth of GPU memory
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_begin_),
                           sizeof(GPUDataT) * total_length_));
            // Ensure the end of the address-range is tracked
            gpu_buffer_end_ = gpu_buffer_begin_ + total_length_;
        }
    }

    DataBufferArena(const std::vector<std::size_t> &buffer_lengths,
                    const DevTag<DevTagT> &dev, bool alloc_memory = true)
        : lengths_{buffer_lengths},
          gpu_buffer_offsets_(buffer_lengths.size() + 1, 0), dev_tag_{dev},
          gpu_buffer_begin_{nullptr}, gpu_buffer_end_{nullptr} {
        std::partial_sum(buffer_lengths.cbegin(), buffer_lengths.cend(),
                         gpu_buffer_offsets_.begin() + 1);
        total_length_ = gpu_buffer_offsets_.back();
        if (alloc_memory && (buffer_lengths.size() > 0) &&
            (buffer_lengths[0] > 0)) {
            dev_tag_.refresh();

            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_begin_),
                           sizeof(GPUDataT) * total_length_));
            gpu_buffer_end_ = gpu_buffer_begin_ + total_length_;
        }
    }

    DataBufferArena(const std::vector<std::size_t> &buffer_lengths,
                    DevTag<DevTagT> &&dev, bool alloc_memory = true)
        : lengths_{buffer_lengths},
          gpu_buffer_offsets_(buffer_lengths.size() + 1, 0), dev_tag_{std::move(
                                                                 dev)},
          gpu_buffer_begin_{nullptr}, gpu_buffer_end_{nullptr} {
        std::partial_sum(buffer_lengths.cbegin(), buffer_lengths.cend(),
                         gpu_buffer_offsets_.begin() + 1);
        total_length_ = gpu_buffer_offsets_.back();
        if (alloc_memory && (buffer_lengths.size() > 0) &&
            (buffer_lengths[0] > 0)) {
            dev_tag_.refresh();

            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_begin_),
                           sizeof(GPUDataT) * total_length_));
            gpu_buffer_end_ = gpu_buffer_begin_ + total_length_;
        }
    }

    // Buffer should never be default initialized
    DataBufferArena() = delete;

    DataBufferArena &operator=(const DataBufferArena &other) {
        if (this != &other) {
            int local_dev_id = -1;
            PL_CUDA_IS_SUCCESS(cudaGetDevice(&local_dev_id));

            lengths_ = other.lengths_;
            total_length_ = other.length_;
            gpu_buffer_offsets_ = other.gpu_buffer_offsets_;

            dev_tag_ =
                DevTag<DevTagT>{local_dev_id, other.dev_tag_.getStreamID()};
            dev_tag_.refresh();

            gpu_buffer_offsets_ = other.gpu_buffer_offsets_;
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_begin_),
                           sizeof(GPUDataT) * total_length_));
            gpu_buffer_end_ = gpu_buffer_begin_ + total_length_;

            CopyGpuDataToGpu(other.gpu_buffer_begin_, other.total_length_);
        }
        return *this;
    }

    DataBufferArena &operator=(DataBufferArena &&other) {
        if (this != &other) {
            int local_dev_id = -1;
            PL_CUDA_IS_SUCCESS(cudaGetDevice(&local_dev_id));
            lengths_ = std::move(other.lengths_);
            total_length_ = other.length_;
            gpu_buffer_offsets_ = std::move(other.gpu_buffer_offsets_);

            if (local_dev_id == other.dev_tag_.getDeviceID()) {
                dev_tag_ = std::move(other.dev_tag_);
                dev_tag_.refresh();
                gpu_buffer_begin_ = other.gpu_buffer_begin_;
                gpu_buffer_end_ = other.gpu_buffer_end_;
            } else {
                dev_tag_ =
                    DevTag<DevTagT>{local_dev_id, other.dev_tag_.getStreamID()};
                dev_tag_.refresh();
                PL_CUDA_IS_SUCCESS(cudaMalloc(
                    reinterpret_cast<void **>(&other.gpu_buffer_begin_),
                    sizeof(GPUDataT) * total_length_));
                gpu_buffer_end_ = gpu_buffer_begin_ + total_length_;
                CopyGpuDataToGpu(other.gpu_buffer_begin_, other.total_length_);

                PL_CUDA_IS_SUCCESS(cudaFree(other.gpu_buffer_begin_));
                other.dev_tag_ = {};
            }
            other.total_length_ = 0;
            other.gpu_buffer_begin_ = nullptr;
            other.gpu_buffer_end_ = nullptr;
        }
        return *this;
    };

    virtual ~DataBufferArena() {
        if (gpu_buffer_begin_ != nullptr) {
            PL_CUDA_IS_SUCCESS(cudaFree(gpu_buffer_begin_));
        }
    };

    /**
     * @brief Get mutable const buffer pointer for the arena buffer.
     *
     * @return GPUDataT*
     */
    inline auto getData() -> GPUDataT * { return gpu_buffer_begin_; }

    /**
     * @brief Get the const buffer pointer for the arena buffer.
     *
     * @return const GPUDataT*
     */
    inline auto getData() const -> const GPUDataT * {
        return gpu_buffer_begin_;
    }

    /**
     * @brief Get the mutable buffer pointer for the region at the given index.
     *
     * @param index Buffer region index.
     * @return GPUDataT*
     */
    inline auto getData(std::size_t index) -> GPUDataT * {
        return gpu_buffer_begin_ + gpu_buffer_offsets_[index];
    }

    /**
     * @brief Get the const buffer pointer for the region at the given index.
     *
     * @param index Buffer region index.
     * @return const GPUDataT*
     */
    inline auto getData(std::size_t index) const -> const GPUDataT * {
        return gpu_buffer_begin_ + gpu_buffer_offsets_[index];
    }

    /**
     * @brief Zero-initialize the GPU buffer.
     *
     */
    void zeroInit() {
        PL_CUDA_IS_SUCCESS(
            cudaMemset(gpu_buffer_begin_, 0, total_length_ * sizeof(GPUDataT)));
    }

    /**
     * @brief Zero-initialize the GPU buffer region at the given index.
     *
     * @param index Buffer region to zero.
     */
    void zeroInit(std::size_t index) {
        PL_CUDA_IS_SUCCESS(
            cudaMemset(getData(index), 0, lengths_[index] * sizeof(GPUDataT)));
    }

    /**
     * @brief Get the total length of the arena buffer, including all
     * partitions.
     *
     * @return auto
     */
    inline auto getLength() const { return total_length_; }

    /**
     * @brief Get the length of the arena buffer at the given index.
     *
     * @param index Arena buffer region index.
     * @return std::size_t
     */
    inline auto getLength(std::size_t index) const { return lengths_[index]; }

    /**
     * @brief Get the length of all arena regions.
     *
     * @return const std::vector<std::size_t>&
     */
    const auto &getBufferLengths() const { return lengths_; }

    /**
     * @brief Get the number of arena partitions.
     *
     * @return std:size_t
     */
    inline auto getNumBuffers() const { return lengths_.size(); }

    /**
     * @brief Get the offsets into each arena region indexed from front to back.
     *
     * @return const std::vector<std::size_t>&
     */
    const auto &getBufferOffsets() const { return gpu_buffer_offsets_; }

    /**
     * @brief Get the CUDA stream for the given object.
     *
     * @return const cudaStream_t&
     */
    inline auto getStream() const -> cudaStream_t {
        return dev_tag_.getStreamID();
    }

    inline auto getDevice() const -> int { return dev_tag_.getDeviceID(); }
    inline auto getDevTag() const -> const DevTag<DevTagT> & {
        return dev_tag_;
    }

    /**
     * @brief Copy data of given length from GPU to this GPU.
     *
     * @param gpu_in GPU data pointer to copy from.
     * @param length Length of buffer region.
     * @param async Use asynchronous copy. Defaults to false.
     */
    void CopyGpuDataToGpu(const GPUDataT *gpu_in, std::size_t length,
                          bool async = false) {
        PL_ABORT_IF_NOT(
            getLength() == length,
            "Sizes do not match for GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (async) {
            PL_CUDA_IS_SUCCESS(
                cudaMemcpyAsync(getData(), gpu_in, sizeof(GPUDataT) * length,
                                cudaMemcpyDeviceToDevice, getStream()));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(getData(), gpu_in,
                                          sizeof(GPUDataT) * length,
                                          cudaMemcpyDefault));
        }
    }

    /**
     * @brief Copy data of given length from GPU to this GPU starting at the
     * given arena index.
     *
     * @param gpu_in GPU data pointer to copy from.
     * @param length Length of buffer region.
     * @param async Use asynchronous copy. Defaults to false.
     * @param index Index into arena partitions.
     */
    void CopyGpuDataToGpu(const GPUDataT *gpu_in, std::size_t length,
                          bool async, std::size_t index) {
        PL_ABORT_IF_NOT(
            getLength(index) == length,
            "Sizes do not match for GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (async) {
            PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                getData(index), gpu_in, sizeof(GPUDataT) * length,
                cudaMemcpyDeviceToDevice, getStream()));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(getData(index), gpu_in,
                                          sizeof(GPUDataT) * length,
                                          cudaMemcpyDefault));
        }
    }

    /**
     * @brief Copy another DataBufferArena to this GPU.
     *
     * @param buffer Other arena buffer to copy.
     * @param async Use asynchronous copy. Defaults to false.
     */
    void CopyGpuDataToGpu(const DataBufferArena &buffer, bool async = false) {
        CopyGpuDataToGpu(buffer.getData(), buffer.getLength(), async);
    }

    /**
     * @brief Copy another DataBufferArena indexed partition to this GPU at the
     * given index.
     *
     * @param buffer Other arena buffer to copy.
     * @param async Use asynchronous copy. Defaults to false.
     * @param src_index Index into source buffer.
     * @param tgt_index Index into target buffer.
     */
    void CopyGpuDataToGpu(const DataBufferArena &buffer, bool async,
                          std::size_t src_index, std::size_t tgt_index) {
        CopyGpuDataToGpu(buffer.getData(src_index), buffer.getLength(src_index),
                         async, tgt_index);
    }

    /**
     * @brief Copy data of given length from host memory to GPU.
     *
     * @tparam HostDataT Host data type. Defaults to be GPUDataT. Aborts if type
     * sizes do not match.
     * @param host_in Host data pointer to copy from.
     * @param length Length of buffer region.
     * @param async Use asynchronous copy. Defaults to false.
     */
    template <class HostDataT = GPUDataT>
    void CopyHostDataToGpu(const HostDataT *host_in, std::size_t length,
                           bool async = false) {
        PL_ABORT_IF_NOT(
            (getLength() * sizeof(GPUDataT)) == (length * sizeof(HostDataT)),
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
     * @brief Copy data of given length from host memory to GPU starting at the
     * given arena index.
     *
     * @tparam HostDataT Host data type. Defaults to be GPUDataT. Aborts if type
     * sizes do not match.
     * @param host_in Host input data buffer.
     * @param length Length of host buffer region.
     * @param async Use asynchronous copy. Defaults to false.
     * @param tgt_index Index into GPU target buffer.
     */
    template <class HostDataT = GPUDataT>
    void CopyHostDataToGpu(const HostDataT *host_in, std::size_t length,
                           bool async, std::size_t tgt_index) {
        PL_ABORT_IF_NOT(
            (getLength() * sizeof(GPUDataT)) == (length * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (async) {
            PL_CUDA_IS_SUCCESS(
                cudaMemcpyAsync(getData(tgt_index), host_in,
                                sizeof(GPUDataT) * getLength(tgt_index),
                                cudaMemcpyHostToDevice, getStream()));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(
                getData(tgt_index), host_in,
                sizeof(GPUDataT) * getLength(tgt_index), cudaMemcpyDefault));
        }
    }

    /**
     * @brief Copy data of given length from GPU buffer to host.
     *
     * @tparam HostDataT Host data type. Defaults to be GPUDataT. Aborts if type
     * sizes do not match.
     * @param host_out Host output data buffer.
     * @param length Length of host buffer region.
     * @param async Use asynchronous copy. Defaults to false.
     */
    template <class HostDataT = GPUDataT>
    inline void CopyGpuDataToHost(HostDataT *host_out, std::size_t length,
                                  bool async = false) const {
        PL_ABORT_IF_NOT(
            (getLength() * sizeof(GPUDataT)) == (length * sizeof(HostDataT)),
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

    /**
     * @brief Copy data of given length from GPU buffer to host.
     *
     * @tparam HostDataT Host data type. Defaults to be GPUDataT. Aborts if type
     * sizes do not match.
     * @param host_out Host output data buffer.
     * @param length Length of host buffer region.
     * @param async Use asynchronous copy. Defaults to false.
     * @param src_index GPU arena buffer source index.
     */
    template <class HostDataT = GPUDataT>
    inline void CopyGpuDataToHost(HostDataT *host_out, std::size_t length,
                                  bool async, std::size_t src_index) const {
        PL_ABORT_IF_NOT(
            (getLength(src_index) * sizeof(GPUDataT)) ==
                (length * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (!async) {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(
                host_out, getData(src_index),
                sizeof(GPUDataT) * getLength(src_index), cudaMemcpyDefault));
        } else {
            PL_CUDA_IS_SUCCESS(
                cudaMemcpyAsync(host_out, getData(src_index),
                                sizeof(GPUDataT) * getLength(src_index),
                                cudaMemcpyDeviceToHost, getStream()));
        }
    }

  private:
    std::vector<std::size_t> lengths_;
    std::size_t total_length_;
    std::vector<std::size_t> gpu_buffer_offsets_;
    DevTag<DevTagT> dev_tag_;
    GPUDataT *gpu_buffer_begin_;
    GPUDataT *gpu_buffer_end_;
};
} // namespace Pennylane::CUDA
