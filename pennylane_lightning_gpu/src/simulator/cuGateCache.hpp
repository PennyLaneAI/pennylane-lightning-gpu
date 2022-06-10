#pragma once

#include <cmath>
#include <complex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Gates.hpp"
#include "cuda.h"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::CUDA::Util;
using namespace cuUtil;

} // namespace
/// @endcond

namespace Pennylane::CUDA {

/**
 * @brief Represents a cache for gate data to be accessible on the device.
 *
 * @tparam fp_t Floating point precision.
 */
template <class fp_t> class GateCache {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(fp_t{}));
    using gate_id = std::pair<std::string, fp_t>;

    GateCache() = delete;
    GateCache(bool populate) : total_alloc_bytes_{0} {
        if (populate) {
            defaultPopulateCache();
        }
    }
    ~GateCache() {
        for (auto &[k, v] : device_gates_) {
            PL_CUDA_IS_SUCCESS(cudaFree(v));
            v = nullptr;
        }
    };

    /**
     * @brief Add a default gate-set to the given cache. Assumes
     * initializer-list evaluated gates for "PauliX", "PauliY", "PauliZ",
     * "Hadamard", "S", "T", "SWAP", with "CNOT" and "CZ" represented as their
     * single-qubit values.
     *
     */
    void defaultPopulateCache() {
        host_gates_[std::make_pair(std::string{"Identity"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>()};
        host_gates_[std::make_pair(std::string{"PauliX"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
                               cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>()};
        host_gates_[std::make_pair(std::string{"PauliY"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ZERO<CFP_t>(), -cuUtil::IMAG<CFP_t>(),
                               cuUtil::IMAG<CFP_t>(), cuUtil::ZERO<CFP_t>()};
        host_gates_[std::make_pair(std::string{"PauliZ"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), -cuUtil::ONE<CFP_t>()};
        host_gates_[std::make_pair(std::string{"Hadamard"}, 0.0)] =
            std::vector<CFP_t>{
                cuUtil::INVSQRT2<CFP_t>(), cuUtil::INVSQRT2<CFP_t>(),
                cuUtil::INVSQRT2<CFP_t>(), -cuUtil::INVSQRT2<CFP_t>()};
        host_gates_[std::make_pair(std::string{"S"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), cuUtil::IMAG<CFP_t>()};
        host_gates_[std::make_pair(std::string{"T"}, 0.0)] = std::vector<CFP_t>{
            cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ConstMultSC(
                cuUtil::SQRT2<fp_t>() / 2.0,
                cuUtil::ConstSum(cuUtil::ONE<CFP_t>(), cuUtil::IMAG<CFP_t>()))};
        host_gates_[std::make_pair(std::string{"SWAP"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                               cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>()};
        host_gates_[std::make_pair(std::string{"CNOT"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
                               cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>()};
        host_gates_[std::make_pair(std::string{"Toffoli"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
                               cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>()};
        host_gates_[std::make_pair(std::string{"CZ"}, 0.0)] =
            std::vector<CFP_t>{cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                               cuUtil::ZERO<CFP_t>(), -cuUtil::ONE<CFP_t>()};
        host_gates_[std::make_pair(std::string{"CSWAP"}, 0.0)] =
            host_gates_.at(std::make_pair(std::string{"SWAP"}, 0.0));
        for (const auto &[h_gate_k, h_gate_v] : host_gates_) {
            device_gates_[h_gate_k] = nullptr;
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&device_gates_[h_gate_k]),
                           sizeof(CFP_t) * h_gate_v.size()));
            CopyHostDataToGpu(h_gate_v, device_gates_[h_gate_k]);
            total_alloc_bytes_ += (sizeof(CFP_t) * h_gate_v.size());
        }
    }

    /**
     * @brief Check for the existence of a given gate.
     *
     * @param gate_id std::pair of gate_name and given parameter value.
     * @return true Gate exists in cache.
     * @return false Gate does not exist in cache.
     */
    bool gateExists(const gate_id &gate) {
        return ((host_gates_.find(gate) != host_gates_.end()) &&
                (device_gates_.find(gate)) != device_gates_.end());
    }
    /**
     * @brief Check for the existence of a given gate.
     *
     * @param gate_name String of gate name.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @return true Gate exists in cache.
     * @return false Gate does not exist in cache.
     */
    bool gateExists(const std::string &gate_name, fp_t gate_param) {
        return (host_gates_.find(std::make_pair(gate_name, gate_param)) !=
                host_gates_.end()) &&
               (device_gates_.find(std::make_pair(gate_name, gate_param)) !=
                device_gates_.end());
    }

    /**
     * @brief Add gate numerical value to the cache, indexed by the gate name
     * and parameter value.
     *
     * @param gate_name String representing the name of the given gate.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @param host_data Vector of the gate values in row-major order.
     */
    void add_gate(const std::string &gate_name, fp_t gate_param,
                  std::vector<CFP_t> host_data) {
        const auto idx = std::make_pair(gate_name, gate_param);
        host_gates_[idx] = std::move(host_data);
        auto &gate = host_gates_[idx];
        device_gates_[idx] = nullptr;

        PL_CUDA_IS_SUCCESS(
            cudaMalloc(reinterpret_cast<void **>(&device_gates_[idx]),
                       sizeof(CFP_t) * gate.size()));

        CopyHostDataToGpu(gate, device_gates_[idx]);

        total_alloc_bytes_ += (sizeof(CFP_t) * gate.size());
    }
    /**
     * @brief see `void add_gate(const std::string &gate_name, fp_t gate_param,
                  const std::vector<CFP_t> &host_data)`
     *
     * @param gate_key
     * @param host_data
     */
    void add_gate(const gate_id &gate_key, std::vector<CFP_t> host_data) {
        host_gates_[gate_key] = std::move(host_data);
        auto &gate = host_gates_[gate_key];
        device_gates_[gate_key] = nullptr;

        PL_CUDA_IS_SUCCESS(
            cudaMalloc(reinterpret_cast<void **>(&device_gates_[gate_key]),
                       sizeof(CFP_t) * gate.size()));
        CopyHostDataToGpu(gate, device_gates_[gate_key]);

        total_alloc_bytes_ += (sizeof(CFP_t) * gate.size());
    }

    /**
     * @brief Returns a pointer to the GPU device memory where the gate is
     * stored.
     *
     * @param gate_name String representing the name of the given gate.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @return CFP_t* Pointer to gate values on device.
     */
    CFP_t *get_gate_device_ptr(const std::string &gate_name, fp_t gate_param) {
        return device_gates_[std::make_pair(gate_name, gate_param)];
    }
    CFP_t *get_gate_device_ptr(const gate_id &gate_key) {
        return device_gates_[gate_key];
    }
    auto get_gate_host(const std::string &gate_name, fp_t gate_param) {
        return host_gates_[std::make_pair(gate_name, gate_param)];
    }
    auto get_gate_host(const gate_id &gate_key) {
        return host_gates_[gate_key];
    }

  private:
    std::size_t total_alloc_bytes_;

    struct gate_id_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &pair) const {
            return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
        }
    };

    std::unordered_map<gate_id, CFP_t *, gate_id_hash> device_gates_;
    std::unordered_map<gate_id, std::vector<CFP_t>, gate_id_hash> host_gates_;

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     * @param host_gate Complex gate data
     * @param device_ptr Pointer to CUDA device memory.
     */
    inline void
    CopyHostDataToGpu(const std::vector<std::complex<fp_t>> &host_gate,
                      CFP_t *device_ptr) {
        PL_CUDA_IS_SUCCESS(
            cudaMemcpy(device_ptr, reinterpret_cast<CFP_t *>(host_gate.data()),
                       sizeof(std::complex<fp_t>) * host_gate.size(),
                       cudaMemcpyHostToDevice));
    }
    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     * @param host_gate Complex gate data
     * @param device_ptr Pointer to CUDA device memory.
     */
    inline void CopyHostDataToGpu(const std::vector<CFP_t> &host_gate,
                                  CFP_t *device_ptr) {
        PL_CUDA_IS_SUCCESS(cudaMemcpy(device_ptr, host_gate.data(),
                                      sizeof(CFP_t) * host_gate.size(),
                                      cudaMemcpyHostToDevice));
    }
};

} // namespace Pennylane::CUDA