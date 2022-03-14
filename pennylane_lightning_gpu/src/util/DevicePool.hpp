#pragma once

#include "cuda.h"
#include <mutex>
#include <unordered_set>
#include <vector>

#include "TSQueue.hpp"
#include "cuda_helpers.hpp"

namespace Pennylane::CUDA {

/** Manages the available GPU devices in a pool of requestable resources.
 */
template <typename DeviceIndexType = int> class DevicePool {
  public:
    DevicePool() {
        for (std::size_t i = 0; i < DevicePool::getTotalDevices(); i++) {
            available_devices_.push(static_cast<DeviceIndexType>(i));
        }
    }
    virtual ~DevicePool() = default;

    /**
     * @brief Get the indices of devices currently active.
     *
     * @return const std::unordered_set<DeviceIndexType>&
     */
    auto getActiveDevices() -> const std::unordered_set<DeviceIndexType> & {
        return active_devices_;
    }

    /**
     * @brief Check if a given device index is active.
     *
     * @param index Device index label.
     * @return true
     * @return false
     */
    inline bool isActive(const DeviceIndexType &index) {
        return active_devices_.find(index) != active_devices_.end();
    }
    /**
     * @brief Check if a given device index is inactive.
     *
     * @param index Device index label.
     * @return true
     * @return false
     */
    inline bool isInactive(const DeviceIndexType &index) {
        return !isActive(index);
    }

    /**
     * @brief Get the total number of available devices.
     *
     * @return std::size_t
     */
    static std::size_t getTotalDevices() {
        return static_cast<std::size_t>(Util::getGPUCount());
    }

    /**
     * @brief Acquire and return the index for an unused device. Returned device
     * index becomes active.
     *
     * @return int
     */
    int acquireDevice() {
        int dev_id;
        available_devices_.wait_and_pop(dev_id);
        {
            std::lock_guard<std::mutex> lg(m_);
            active_devices_.insert(dev_id);
        }
        return dev_id;
    }

    /**
     * @brief Deactivate given device index, and return to pool.
     *
     * @param dev_id
     */
    void releaseDevice(DeviceIndexType dev_id) {
        available_devices_.push(dev_id);
        {
            std::lock_guard<std::mutex> lg(m_);
            active_devices_.erase(dev_id);
        }
    }

    /**
     * @brief Get the UIDs of available devices.
     *
     * @return std::vector<cudaUUID_t>
     */
    static std::vector<cudaUUID_t> getDeviceUIDs() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        int device;
        std::vector<cudaUUID_t> dev_uid(deviceCount);
        for (device = 0; device < deviceCount; device++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            dev_uid[device] = deviceProp.uuid;
        }
        return dev_uid;
    }

  private:
    std::unordered_set<DeviceIndexType> active_devices_;
    std::mutex m_;
    TSQueue<DeviceIndexType> available_devices_;
};

} // namespace Pennylane::CUDA