#pragma once

#include "cuda.h"
#include "cuda_helpers.hpp"

namespace Pennylane::CUDA {

/**
 * @brief Utility class to hold device ID and associated stream ID. Data are
 * assumed immutable upon creation.
 *
 */
template <class IDType = int>
class DevID {
  public:
    DevID() : device_id_{0}, stream_id_{0} {}
    DevID(int device_id, cudaStream_t stream_id) {}
    ~virtual DevID() {}

    auto getDeviceID() const -> int { return device_id; }
    auto getStreamID() const -> cudaStream_t { return stream_id_; }

    inline bool operator==(const DevID &other) {
        return (getDeviceID() == other.getDeviceID()) &&
               (getStreamID() == other.getStreamID());
    }

  private:
    const IDType device_id_;
    const cudaStream_t stream_id_;
};


} // namespace Pennylane::CUDA