#pragma once

#include "DevTag.hpp"
#include "cuda.h"
#include "cuda_helpers.hpp"

namespace Pennylane::CUDA {

template <class GPUDataT, class DevTagT = int> class DataBufferArena {
  private:
    std::vector<std::size_t> lengths_;
    std::size_t total_length_;
    std::vector<std::size_t> gpu_buffer_offsets_;
    DevTag<DevTagT> dev_tag_;
    GPUDataT *gpu_buffer_begin_;
    GPUDataT *gpu_buffer_end_;

  public:
    using type = GPUDataT;

    ///////////////////////////////////////////////////////////////////////////
    //
    ///////////////////////////////////////////////////////////////////////////

    DataBufferArena(const std::vector<std::size_t> &buffer_lengths,
                    int device_id = 0, cudaStream_t stream_id = 0,
                    bool alloc_memory = true)
        : lengths_{buffer_lengths}, dev_tag_{device_id, stream_id},
          gpu_buffer_begin_{nullptr}, gpu_buffer_end_{nullptr} {

        if (alloc_memory && (buffer_lengths.size() > 0)) {
            // Ensure we tag the current GPU
            dev_tag_.refresh();
            // Create enough room for the offset locations
            gpu_buffer_offsets_.reserve(buffer_lengths.size() + 1);
            gpu_buffer_offsets_[0] = 0;
            // Define the pointer offsets for each buffer
            std::partial_sum(buffer_lengths.cbegin(), buffer_lengths.cend(),
                             buffer_lengths.begin() + 1);
            total_length_ = gpu_buffer_offsets_.back();
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
        : lengths_{buffer_lengths}, dev_tag_{dev}, gpu_buffer_begin_{nullptr},
          gpu_buffer_end_{nullptr} {
        if (alloc_memory && (buffer_lengths.size() > 0)) {
            dev_tag_.refresh();
            gpu_buffer_offsets_.reserve(buffer_lengths.size() + 1);
            gpu_buffer_offsets_[0] = 0;
            std::partial_sum(buffer_lengths.cbegin(), buffer_lengths.cend(),
                             buffer_lengths.begin() + 1);
            total_length_ = gpu_buffer_offsets_.back();

            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_begin_),
                           sizeof(GPUDataT) * total_length_));
            gpu_buffer_end_ = gpu_buffer_begin_ + total_length_;
        }
    }

    DataBufferArena(const std::vector<std::size_t> &buffer_lengths,
                    DevTag<DevTagT> &&dev, bool alloc_memory = true)
        : lengths_{buffer_lengths}, dev_tag_{std::move(dev)},
          gpu_buffer_begin_{nullptr}, gpu_buffer_end_{nullptr} {
        if (alloc_memory && (buffer_lengths.size() > 0)) {
            dev_tag_.refresh();
            gpu_buffer_offsets_.reserve(buffer_lengths.size() + 1);
            gpu_buffer_offsets_[0] = 0;
            std::partial_sum(buffer_lengths.cbegin(), buffer_lengths.cend(),
                             buffer_lengths.begin() + 1);
            total_length_ = gpu_buffer_offsets_.back();

            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_begin_),
                           sizeof(GPUDataT) * total_length_));
            gpu_buffer_end_ = gpu_buffer_begin_ + total_length_;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    ///////////////////////////////////////////////////////////////////////////

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

    auto getData() -> GPUDataT * { return gpu_buffer_begin_; }
    auto getData() const -> const GPUDataT * { return gpu_buffer_begin_; }

    auto getData(std::size_t index) -> GPUDataT * {
        return gpu_buffer_begin_ + gpu_buffer_offsets_[index];
    }
    auto getData(std::size_t buffer_index) const -> const GPUDataT * {
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
    void zeroInit(std::size_t index) {
        PL_CUDA_IS_SUCCESS(
            cudaMemset(getData(index), 0, lengths_[index] * sizeof(GPUDataT)));
    }

    auto getLength() const { return total_length_; }
    auto getLength(std::size_t index) const { return lengths_[index]; }

    auto &getBufferLengths() const { return lengths_; }
    auto getNumBuffers() const { return lengths_.size(); }
    auto &getBufferOffsets() const { return gpu_buffer_offsets_; }

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
     * @brief Copy data from another GPU memory block to here.
     *
     */
    void CopyGpuDataToGpu(const GPUDataT *gpu_in, std::size_t length,
                          bool async = false, std::size_t index = 0) {
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
     * @brief Copy data from another GPU memory block to here.
     *
     */
    void CopyGpuDataToGpu(const DataBufferArena &buffer, bool async = false) {
        CopyGpuDataToGpu(buffer.getData(), buffer.getLengthTotal(), async);
    }

    void CopyGpuDataToGpu(const DataBufferArena &buffer, bool async = false,
                          std::size_t index = 0) {
        CopyGpuDataToGpu(buffer.getData(index), buffer.getLength(index), async);
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
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

    template <class HostDataT = GPUDataT>
    void CopyHostDataToGpu(const HostDataT *host_in, std::size_t length,
                           bool async = false, std::size_t index = 0) {
        PL_ABORT_IF_NOT(
            (getLength() * sizeof(GPUDataT)) == (length * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (async) {
            PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                getData(index), host_in, sizeof(GPUDataT) * getLength(index),
                cudaMemcpyHostToDevice, getStream()));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(getData(index), host_in,
                                          sizeof(GPUDataT) * getLength(index),
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
    template <class HostDataT = GPUDataT>
    inline void CopyGpuDataToHost(HostDataT *host_out, std::size_t length,
                                  bool async = false,
                                  std::size_t index = 0) const {
        PL_ABORT_IF_NOT(
            (getLength(index) * sizeof(GPUDataT)) ==
                (length * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (!async) {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(host_out, getData(index),
                                          sizeof(GPUDataT) * getLength(index),
                                          cudaMemcpyDefault));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                host_out, getData(index), sizeof(GPUDataT) * getLength(index),
                cudaMemcpyDeviceToHost, getStream()));
        }
    }
};
} // namespace Pennylane::CUDA
