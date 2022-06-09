#pragma once

#include "cuda.h"
#include "cuda_helpers.hpp"
#include <type_traits>

namespace Pennylane::CUDA {

/**
 * @brief Utility class to hold device ID and associated stream ID.
 *
 */
template <class IDType = int,
          std::enable_if_t<std::is_assignable<IDType &, int>::value, bool> =
              true>
class DevTag {
  public:
    DevTag() : device_id_{0}, stream_id_{0} {}

    DevTag(IDType device_id, cudaStream_t stream_id)
        : device_id_{device_id}, stream_id_{stream_id} {}

    DevTag(const DevTag &other)
        : device_id_{other.getDeviceID()}, stream_id_{other.getStreamID()} {}

    DevTag(DevTag &&other) noexcept
        : device_id_{std::move(other.device_id_)}, stream_id_{std::move(
                                                       other.stream_id_)} {
        auto ref_id = &other.device_id_;
        auto ref_st = &other.stream_id_;
        ref_id = nullptr;
        ref_st = nullptr;
    }

    DevTag &operator=(DevTag &&other) {
        if (this != &other) {
            device_id_ = other.device_id_;
            stream_id_ = other.stream_id_;
            auto ref_id = &other.device_id_;
            auto ref_st = &other.stream_id_;
            ref_id = nullptr;
            ref_st = nullptr;
            static_cast<void>(ref_id);
            static_cast<void>(ref_st);
        }
        return *this;
    }

    virtual ~DevTag() {}

    auto getDeviceID() const -> IDType { return device_id_; }
    auto getStreamID() const -> cudaStream_t { return stream_id_; }

    inline bool operator==(const DevTag &other) {
        return (getDeviceID() == other.getDeviceID()) &&
               (getStreamID() == other.getStreamID());
    }

    inline void refresh(){
        PL_CUDA_IS_SUCCESS(cudaSetDevice(device_id_));
    }

  private:
    IDType device_id_;
    cudaStream_t stream_id_;
};

} // namespace Pennylane::CUDA