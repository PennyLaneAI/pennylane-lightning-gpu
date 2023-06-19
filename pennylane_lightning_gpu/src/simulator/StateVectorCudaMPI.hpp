// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file StateVectorCudaMPI.hpp
 */
#pragma once

#include <functional>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuComplex.h> // cuDoubleComplex
#include <cuda.h>
#include <custatevec.h> // custatevecApplyMatrix

#include "Constant.hpp"
#include "Error.hpp"
#include "MPIManager.hpp"
#include "MPIWorker.hpp"
#include "StateVectorCudaBase.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::CUDA::Util;
using namespace Pennylane::CUDA;
using namespace Pennylane::MPI;
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane {

// declarations of external functions (defined in initSV.cu).
extern void setStateVector_CUDA(cuComplex *sv, int &num_indices,
                                cuComplex *value, int *indices,
                                size_t thread_per_block,
                                cudaStream_t stream_id);
extern void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                                cuDoubleComplex *value, long *indices,
                                size_t thread_per_block,
                                cudaStream_t stream_id);

extern void setBasisState_CUDA(cuComplex *sv, cuComplex &value,
                               const size_t index, bool async,
                               cudaStream_t stream_id);
extern void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                               const size_t index, bool async,
                               cudaStream_t stream_id);

/**
 * @brief Managed memory CUDA state-vector class using custateVec backed
 * gate-calls.
 *
 * @tparam Precision Floating-point precision type.
 */
template <class Precision>
class StateVectorCudaMPI
    : public StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>> {
  private:
    using BaseType = StateVectorCudaBase<Precision, StateVectorCudaMPI>;

    size_t numGlobalQubits_;
    size_t numLocalQubits_;
    MPIManager mpi_manager_;

    SharedCusvHandle handle_;
    SharedCublasCaller cublascaller_;
    mutable SharedCusparseHandle
        cusparsehandle_; // This member is mutable to allow lazy initialization.
    SharedLocalStream localStream_;
    SharedMPIWorker svSegSwapWorker_;
    GateCache<Precision> gate_cache_;

  public:
    using CFP_t =
        typename StateVectorCudaBase<Precision,
                                     StateVectorCudaMPI<Precision>>::CFP_t;
    using GateType = CFP_t *;

    StateVectorCudaMPI() = delete;

    StateVectorCudaMPI(MPIManager mpi_manager, const DevTag<int> &dev_tag,
                       size_t mpi_buf_size, size_t num_global_qubits,
                       size_t num_local_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(mpi_manager),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, mpi_buf_size, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    };

    StateVectorCudaMPI(MPI_Comm mpi_communicator, const DevTag<int> &dev_tag,
                       size_t mpi_buf_size, size_t num_global_qubits,
                       size_t num_local_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(mpi_communicator),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, mpi_buf_size, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    };

    StateVectorCudaMPI(const DevTag<int> &dev_tag, size_t mpi_buf_size,
                       size_t num_global_qubits, size_t num_local_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(MPI_COMM_WORLD),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, mpi_buf_size, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    };

    StateVectorCudaMPI(const DevTag<int> &dev_tag, size_t num_global_qubits,
                       size_t num_local_qubits, const CFP_t *gpu_data)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(MPI_COMM_WORLD),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, 0, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        size_t length = 1 << numLocalQubits_;
        BaseType::CopyGpuDataToGpuIn(gpu_data, length, false);
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize())
        mpi_manager_.Barrier();
    }

    StateVectorCudaMPI(const DevTag<int> &dev_tag, size_t num_global_qubits,
                       size_t num_local_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(MPI_COMM_WORLD),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, 0, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        initSV_MPI();
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    ~StateVectorCudaMPI() = default;

    /**
     * @brief Get MPI manager
     */
    auto getMPIManager() const { return mpi_manager_; }

    /**
     * @brief Get the total number of wires.
     */
    auto getTotalNumQubits() const -> size_t {
        return numGlobalQubits_ + numLocalQubits_;
    }

    /**
     * @brief Get the number of wires distributed across devices.
     */
    auto getNumGlobalQubits() const -> size_t { return numGlobalQubits_; }

    /**
     * @brief Get the number of wires within the local devices.
     */
    auto getNumLocalQubits() const -> size_t { return numLocalQubits_; }

    /**
     * @brief Get pointer to custatevecSVSwapWorkerDescriptor.
     */
    auto getSwapWorker() -> custatevecSVSwapWorkerDescriptor_t {
        return svSegSwapWorker_.get();
    }
    /**
     * @brief Init 00....0>.
     */
    void initSV_MPI(bool async = false) {
        size_t index = 0;
        const std::complex<Precision> value = {1, 0};
        BaseType::getDataBuffer().zeroInit();
        setBasisState(value, index, async);
    }

    /**
     * @brief Set value for a single element of the state-vector on device. This
     * method is implemented by cudaMemcpy.
     *
     * @param value Value to be set for the target element.
     * @param index Index of the target element.
     * @param async Use an asynchronous memory copy.
     */
    void setBasisState(const std::complex<Precision> &value, const size_t index,
                       const bool async = false) {
        size_t rankId = index >> BaseType::getNumQubits();

        size_t local_index =
            static_cast<size_t>(rankId *
                                std::pow(2.0, static_cast<long double>(
                                                  BaseType::getNumQubits()))) ^
            index;
        BaseType::getDataBuffer().zeroInit();

        CFP_t value_cu = cuUtil::complexToCu<std::complex<Precision>>(value);
        auto stream_id = localStream_.get();

        if (mpi_manager_.getRank() == rankId) {
            setBasisState_CUDA(BaseType::getData(), value_cu, local_index,
                               async, stream_id);
        }
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    /**
     * @brief Set values for a batch of elements of the state-vector. This
     * method is implemented by the customized CUDA kernel defined in the
     * DataBuffer class.
     *
     * @param num_indices Number of elements to be passed to the state vector.
     * @param values Pointer to values to be set for the target elements.
     * @param indices Pointer to indices of the target elements.
     * @param async Use an asynchronous memory copy.
     */
    template <class index_type, size_t thread_per_block = 256>
    void setStateVector(const index_type num_indices,
                        const std::complex<Precision> *values,
                        const index_type *indices, const bool async = false) {

        BaseType::getDataBuffer().zeroInit();

        std::vector<index_type> indices_local;
        std::vector<std::complex<Precision>> values_local;

        for (size_t i = 0; i < static_cast<size_t>(num_indices); i++) {
            int index = indices[i];
            assert(index >= 0);
            size_t rankId =
                static_cast<size_t>(index) >> BaseType::getNumQubits();

            if (rankId == mpi_manager_.getRank()) {
                int local_index =
                    static_cast<size_t>(
                        rankId * std::pow(2.0, static_cast<long double>(
                                                   BaseType::getNumQubits()))) ^
                    index;
                indices_local.push_back(local_index);
                values_local.push_back(values[i]);
            }
        }

        auto device_id = BaseType::getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = BaseType::getDataBuffer().getDevTag().getStreamID();

        index_type num_elements = indices_local.size();

        DataBuffer<index_type, int> d_indices{
            static_cast<std::size_t>(num_elements), device_id, stream_id, true};

        DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(num_elements),
                                        device_id, stream_id, true};

        d_indices.CopyHostDataToGpu(indices_local.data(), d_indices.getLength(),
                                    async);
        d_values.CopyHostDataToGpu(values_local.data(), d_values.getLength(),
                                   async);

        setStateVector_CUDA(BaseType::getData(), num_elements,
                            d_values.getData(), d_indices.getData(),
                            thread_per_block, stream_id);
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    /**
     * @brief Apply a single gate to the state-vector. Offloads to custatevec
     * specific API calls if available. If unable, attempts to use prior cached
     * gate values on the device. Lastly, accepts a host-provided matrix if
     * otherwise, and caches on the device for later reuse.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Matrix representation of gate.
     */
    void applyOperation(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<CFP_t> &gate_matrix = {}) {
        const auto ctrl_offset = (BaseType::getCtrlMap().find(opName) !=
                                  BaseType::getCtrlMap().end())
                                     ? BaseType::getCtrlMap().at(opName)
                                     : 0;
        const std::vector<std::size_t> ctrls{wires.begin(),
                                             wires.begin() + ctrl_offset};
        const std::vector<std::size_t> tgts{wires.begin() + ctrl_offset,
                                            wires.end()};
        if (opName == "Identity") {
            return;
        } else if (native_gates_.find(opName) != native_gates_.end()) {
            applyParametricPauliGate({opName}, ctrls, tgts, params.front(),
                                     adjoint);
        } else if (opName == "Rot" || opName == "CRot") {
            if (adjoint) {
                applyParametricPauliGate({"RZ"}, ctrls, tgts, params[2], true);
                applyParametricPauliGate({"RY"}, ctrls, tgts, params[1], true);
                applyParametricPauliGate({"RZ"}, ctrls, tgts, params[0], true);
            } else {
                applyParametricPauliGate({"RZ"}, ctrls, tgts, params[0], false);
                applyParametricPauliGate({"RY"}, ctrls, tgts, params[1], false);
                applyParametricPauliGate({"RZ"}, ctrls, tgts, params[2], false);
            }
        } else if (par_gates_.find(opName) != par_gates_.end()) {
            par_gates_.at(opName)(wires, adjoint, params);
        } else { // No offloadable function call; defer to matrix passing
            auto &&par =
                (params.empty()) ? std::vector<Precision>{0.0} : params;
            // ensure wire indexing correctly preserved for tensor-observables
            const std::vector<std::size_t> ctrls_local{ctrls.rbegin(),
                                                       ctrls.rend()};
            const std::vector<std::size_t> tgts_local{tgts.rbegin(),
                                                      tgts.rend()};

            if (!gate_cache_.gateExists(opName, par[0]) &&
                gate_matrix.empty()) {
                std::string message = "Currently unsupported gate: " + opName;
                throw LightningException(message);
            } else if (!gate_cache_.gateExists(opName, par[0])) {
                gate_cache_.add_gate(opName, par[0], gate_matrix);
            }
            applyDeviceMatrixGate(
                gate_cache_.get_gate_device_ptr(opName, par[0]), ctrls_local,
                tgts_local, adjoint);
        }
    }
    /**
     * @brief STL-friendly variant of `applyOperation(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<CFP_t> &gate_matrix = {})`
     *
     */
    void applyOperation_std(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<std::complex<Precision>>
            &gate_matrix = {}) {
        std::vector<CFP_t> matrix_cu(gate_matrix.size());
        std::transform(gate_matrix.begin(), gate_matrix.end(),
                       matrix_cu.begin(), [](const std::complex<Precision> &x) {
                           return cuUtil::complexToCu<std::complex<Precision>>(
                               x);
                       });
        applyOperation(opName, wires, adjoint, params, matrix_cu);
    }

    /**
     * @brief Multi-op variant of `execute(const std::string &opName, const
     std::vector<int> &wires, bool adjoint = false, const std::vector<Precision>
     &params)`
     *
     * @param opNames
     * @param wires
     * @param adjoints
     * @param params
     */
    void applyOperation(const std::vector<std::string> &opNames,
                        const std::vector<std::vector<size_t>> &wires,
                        const std::vector<bool> &adjoints,
                        const std::vector<std::vector<Precision>> &params) {
        PL_ABORT_IF(opNames.size() != wires.size(),
                    "Incompatible number of ops and wires");
        PL_ABORT_IF(opNames.size() != adjoints.size(),
                    "Incompatible number of ops and adjoints");
        const auto num_ops = opNames.size();
        for (std::size_t op_idx = 0; op_idx < num_ops; op_idx++) {
            applyOperation(opNames[op_idx], wires[op_idx], adjoints[op_idx],
                           params[op_idx]);
        }
    }

    /**
     * @brief Multi-op variant of `execute(const std::string &opName, const
     std::vector<int> &wires, bool adjoint = false, const std::vector<Precision>
     &params)`
     *
     * @param opNames
     * @param wires
     * @param adjoints
     */
    void applyOperation(const std::vector<std::string> &opNames,
                        const std::vector<std::vector<size_t>> &wires,
                        const std::vector<bool> &adjoints) {
        PL_ABORT_IF(opNames.size() != wires.size(),
                    "Incompatible number of ops and wires");
        PL_ABORT_IF(opNames.size() != adjoints.size(),
                    "Incompatible number of ops and adjoints");
        const auto num_ops = opNames.size();
        for (std::size_t op_idx = 0; op_idx < num_ops; op_idx++) {
            applyOperation(opNames[op_idx], wires[op_idx], adjoints[op_idx]);
        }
    }

    //****************************************************************************//
    // Explicit gate calls for bindings
    //****************************************************************************//
    /* one-qubit gates */
    inline void applyIdentity(const std::vector<std::size_t> &wires,
                              bool adjoint) {
        static_cast<void>(wires);
        static_cast<void>(adjoint);
    }
    inline void applyPauliX(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliX"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyPauliY(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliY"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyPauliZ(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliZ"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyHadamard(const std::vector<std::size_t> &wires,
                              bool adjoint) {
        static const std::string name{"Hadamard"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyS(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"S"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyT(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"T"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyRX(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RX"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRY(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RY"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRZ(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RZ"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRot(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param0, Precision param1, Precision param2) {
        if (!adjoint) {
            applyRZ(wires, false, param0);
            applyRY(wires, false, param1);
            applyRZ(wires, false, param2);
        } else {
            applyRZ(wires, true, param2);
            applyRY(wires, true, param1);
            applyRZ(wires, true, param0);
        }
    }
    inline void applyRot(const std::vector<std::size_t> &wires, bool adjoint,
                         const std::vector<Precision> &params) {
        applyRot(wires, adjoint, params[0], params[1], params[2]);
    }
    inline void applyPhaseShift(const std::vector<std::size_t> &wires,
                                bool adjoint, Precision param) {
        static const std::string name{"PhaseShift"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getPhaseShift<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }

    /* two-qubit gates */
    inline void applyCNOT(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CNOT"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CY"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CZ"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applySWAP(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"SWAP"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param), {},
                              wires, adjoint);
    }
    inline void applyIsingXX(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RX"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyIsingYY(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RY"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyIsingZZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RZ"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyCRot(const std::vector<std::size_t> &wires, bool adjoint,
                          const std::vector<Precision> &params) {
        applyCRot(wires, adjoint, params[0], params[1], params[2]);
    }
    inline void applyCRot(const std::vector<std::size_t> &wires, bool adjoint,
                          Precision param0, Precision param1,
                          Precision param2) {
        if (!adjoint) {
            applyCRZ(wires, false, param0);
            applyCRY(wires, false, param1);
            applyCRZ(wires, false, param2);
        } else {
            applyCRZ(wires, true, param2);
            applyCRY(wires, true, param1);
            applyCRZ(wires, true, param0);
        }
    }

    inline void applyCRX(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRX(wires, adjoint, param);
    }
    inline void applyCRY(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRY(wires, adjoint, param);
    }
    inline void applyCRZ(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRZ(wires, adjoint, param);
    }
    inline void applyControlledPhaseShift(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        applyPhaseShift(wires, adjoint, param);
    }
    inline void applySingleExcitation(const std::vector<std::size_t> &wires,
                                      bool adjoint, Precision param) {
        static const std::string name{"SingleExcitation"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getSingleExcitation<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void
    applySingleExcitationMinus(const std::vector<std::size_t> &wires,
                               bool adjoint, Precision param) {
        static const std::string name{"SingleExcitationMinus"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getSingleExcitationMinus<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void applySingleExcitationPlus(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        static const std::string name{"SingleExcitationPlus"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getSingleExcitationPlus<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }

    /* three-qubit gates */
    inline void applyToffoli(const std::vector<std::size_t> &wires,
                             bool adjoint) {
        static const std::string name{"Toffoli"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCSWAP(const std::vector<std::size_t> &wires,
                           bool adjoint) {
        static const std::string name{"SWAP"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.front()}, {wires.begin() + 1, wires.end()},
                              adjoint);
    }

    /* four-qubit gates */
    inline void applyDoubleExcitation(const std::vector<std::size_t> &wires,
                                      bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitation<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }
    inline void
    applyDoubleExcitationMinus(const std::vector<std::size_t> &wires,
                               bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitationMinus<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }
    inline void applyDoubleExcitationPlus(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitationPlus<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }

    /* Multi-qubit gates */
    inline void applyMultiRZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        const std::vector<std::string> names(wires.size(), {"RZ"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }

    /* Gate generators */
    inline void applyGeneratorIsingXX(const std::vector<std::size_t> &wires,
                                      bool adjoint) {
        static const std::string name{"GeneratorIsingXX"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingXX<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void applyGeneratorIsingYY(const std::vector<std::size_t> &wires,
                                      bool adjoint) {
        static const std::string name{"GeneratorIsingYY"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingYY<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void applyGeneratorIsingZZ(const std::vector<std::size_t> &wires,
                                      bool adjoint) {
        static const std::string name{"GeneratorIsingZZ"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingZZ<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }

    inline void
    applyGeneratorSingleExcitation(const std::vector<std::size_t> &wires,
                                   bool adjoint) {
        static const std::string name{"GeneratorSingleExcitation"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitation<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void
    applyGeneratorSingleExcitationMinus(const std::vector<std::size_t> &wires,
                                        bool adjoint) {
        static const std::string name{"GeneratorSingleExcitationMinus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitationMinus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void
    applyGeneratorSingleExcitationPlus(const std::vector<std::size_t> &wires,
                                       bool adjoint) {
        static const std::string name{"GeneratorSingleExcitationPlus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitationPlus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }

    inline void
    applyGeneratorDoubleExcitation(const std::vector<std::size_t> &wires,
                                   bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitation"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitation<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void
    applyGeneratorDoubleExcitationMinus(const std::vector<std::size_t> &wires,
                                        bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitationMinus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitationMinus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void
    applyGeneratorDoubleExcitationPlus(const std::vector<std::size_t> &wires,
                                       bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitationPlus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitationPlus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }

    inline void applyGeneratorMultiRZ(const std::vector<std::size_t> &wires,
                                      bool adjoint) {
        static const std::string name{"PauliZ"};
        static const Precision param = 0.0;
        for (const auto &w : wires) {
            applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                                  {}, {w}, adjoint);
        }
    }

    /**
     * @brief Access the CublasCaller the object is using.
     *
     * @return a reference to the object's CublasCaller object.
     */
    auto getCublasCaller() const -> const CublasCaller & {
        return *cublascaller_;
    }

    /**
     * @brief Get the cuSPARSE handle that the object is using.
     *
     * @return cusparseHandle_t returns the cuSPARSE handle.
     */
    auto getCusparseHandle() const -> cusparseHandle_t {
        if (!cusparsehandle_)
            cusparsehandle_ = make_shared_cusparse_handle();
        return cusparsehandle_.get();
    }

    /**
     * @brief Utility method for expectation value calculations.
     *
     * @param obsName String label for observable. If already exists, will used
     * cached device value. If not, `gate_matrix` is expected, and will
     * automatically cache for future reuse.
     * @param wires Target wires for expectation value.
     * @param params Parameters for a parametric gate.
     * @param gate_matrix Optional matrix for observable. Caches for future use
     * if does not exist.
     * @return auto Expectation value.
     */
    auto expval(const std::string &obsName, const std::vector<size_t> &wires,
                const std::vector<Precision> &params = {0.0},
                const std::vector<CFP_t> &gate_matrix = {}) {
        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;
        auto &&local_wires =
            (gate_matrix.empty())
                ? wires
                : std::vector<size_t>{
                      wires.rbegin(),
                      wires.rend()}; // ensure wire indexing correctly preserved
                                     // for tensor-observables

        if (!(gate_cache_.gateExists(obsName, par[0]) || gate_matrix.empty())) {
            gate_cache_.add_gate(obsName, par[0], gate_matrix);
        } else if (!gate_cache_.gateExists(obsName, par[0]) &&
                   gate_matrix.empty()) {
            std::string message =
                "Currently unsupported observable: " + obsName;
            throw LightningException(message.c_str());
        }
        auto expect_val = getExpectationValueDeviceMatrix(
            gate_cache_.get_gate_device_ptr(obsName, par[0]), local_wires);
        return expect_val;
    }
    /**
     * @brief See `expval(const std::string &obsName, const std::vector<size_t>
     &wires, const std::vector<Precision> &params = {0.0}, const
     std::vector<CFP_t> &gate_matrix = {})`
     */
    auto expval(const std::string &obsName, const std::vector<size_t> &wires,
                const std::vector<Precision> &params = {0.0},
                const std::vector<std::complex<Precision>> &gate_matrix = {}) {
        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;

        std::vector<CFP_t> matrix_cu(gate_matrix.size());
        if (!(gate_cache_.gateExists(obsName, par[0]) || gate_matrix.empty())) {
            for (std::size_t i = 0; i < gate_matrix.size(); i++) {
                matrix_cu[i] = cuUtil::complexToCu<std::complex<Precision>>(
                    gate_matrix[i]);
            }
            gate_cache_.add_gate(obsName, par[0], matrix_cu);
        } else if (!gate_cache_.gateExists(obsName, par[0]) &&
                   gate_matrix.empty()) {
            std::string message =
                "Currently unsupported observable: " + obsName;
            throw LightningException(message.c_str());
        }
        return expval(obsName, wires, params, matrix_cu);
    }
    /**
     * @brief See `expval(std::vector<CFP_t> &gate_matrix = {})`
     */
    auto expval(const std::vector<size_t> &wires,
                const std::vector<std::complex<Precision>> &gate_matrix) {
        std::vector<CFP_t> matrix_cu(gate_matrix.size());

        for (std::size_t i = 0; i < gate_matrix.size(); i++) {
            matrix_cu[i] =
                cuUtil::complexToCu<std::complex<Precision>>(gate_matrix[i]);
        }

        if (gate_matrix.empty()) {
            std::string message = "Currently unsupported observable";
            throw LightningException(message.c_str());
        }

        // Wire order reversed to match expected custatevec wire ordering for
        // tensor observables.
        auto &&local_wires =
            (gate_matrix.empty())
                ? wires
                : std::vector<size_t>{wires.rbegin(), wires.rend()};
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        auto expect_val =
            getExpectationValueDeviceMatrix(matrix_cu.data(), local_wires);
        return expect_val;
    }

    /**
     * @brief Utility method for probability calculation using given wires.
     *
     * @param wires List of wires to return probabilities for in lexicographical
     * order.
     * @return std::vector<double>
     */
    auto probability(const std::vector<size_t> &wires) -> std::vector<double> {
        // Data return type fixed as double in custatevec function call
        std::vector<double> subgroup_probabilities;

        // this should be built upon by the wires not participating
        int maskLen = 0;
        int *maskBitString = nullptr;
        int *maskOrdering = nullptr;

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::vector<int> wires_int(wires.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(
            wires.begin(), wires.end(), wires_int.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // split wires_int to global and local ones
        std::vector<int> wires_local;
        std::vector<int> wires_global;

        for (const auto &wire : wires_int) {
            if (wire < static_cast<int>(this->getNumLocalQubits())) {
                wires_local.push_back(wire);
            } else {
                wires_global.push_back(wire);
            }
        }

        std::vector<double> local_probabilities(Util::exp2(wires_local.size()));

        PL_CUSTATEVEC_IS_SUCCESS(custatevecAbs2SumArray(
            /* custatevecHandle_t */ handle_.get(),
            /* const void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ this->getNumLocalQubits(),
            /* double* */ local_probabilities.data(),
            /* const int32_t* */ wires_local.data(),
            /* const uint32_t */ wires_local.size(),
            /* const int32_t* */ maskBitString,
            /* const int32_t* */ maskOrdering,
            /* const uint32_t */ maskLen));

        // create new MPI communicator groups
        size_t subCommGroupId = 0;
        for (size_t i = 0; i < wires_global.size(); i++) {
            size_t mask = 1 << (wires_global[i] - this->getNumLocalQubits());
            size_t bitValue = mpi_manager_.getRank() & mask;
            subCommGroupId += bitValue
                              << (wires_global[i] - this->getNumLocalQubits());
        }
        auto sub_mpi_manager0 =
            mpi_manager_.split(subCommGroupId, mpi_manager_.getRank());

        if (sub_mpi_manager0.getSize() == 1) {
            return local_probabilities;
        } else {
            if (sub_mpi_manager0.getRank() == 0) {
                subgroup_probabilities.resize(Util::exp2(wires_local.size()));
            }

            sub_mpi_manager0.Reduce<double>(local_probabilities,
                                            subgroup_probabilities, 0, "sum");

            return subgroup_probabilities;
        }
    }

    /**
     * @brief Get expectation value for a sum of Pauli words.
     *
     * @param pauli_words Vector of Pauli-words to evaluate expectation value.
     * @param tgts Coupled qubit index to apply each Pauli term.
     * @param coeffs Numpy array buffer of size |pauli_words|
     * @return auto Expectation value.
     */
    auto getExpectationValuePauliWords(
        const std::vector<std::string> &pauli_words,
        const std::vector<std::vector<std::size_t>> &tgts,
        const std::complex<Precision> *coeffs) {

        enum WiresSwapStatus { Local, Swappable, UnSwappable };

        std::vector<double> expect_local(pauli_words.size());

        std::vector<std::size_t> tgtsSwapStatus;
        std::vector<std::vector<int2>> tgtswirePairs;
        std::vector<std::vector<size_t>> tgtsInC;
        tgtsInC.reserve(tgts.size());

        for (const auto &vec : tgts) {
            std::vector<size_t> tmpVecInt(
                vec.size()); // Reserve memory for efficiency

            std::transform(vec.begin(), vec.end(), tmpVecInt.begin(),
                           [&](std::size_t x) {
                               return this->getTotalNumQubits() - 1 - x;
                           });
            tgtsInC.push_back(std::move(tmpVecInt));
        }

        std::vector<std::vector<size_t>> localTgtsInC;
        localTgtsInC.reserve(tgts.size());

        for (const auto &vec : tgtsInC) {
            std::vector<int> statusWires(this->getTotalNumQubits(),
                                         WireStatus::Default);

            for (size_t i = 0; i < vec.size(); i++) {
                statusWires[vec[i]] = WireStatus::Target;
            }
            size_t StatusGlobalWires =
                std::reduce(statusWires.begin() + this->getNumLocalQubits(),
                            statusWires.end());

            if (!StatusGlobalWires) {
                tgtsSwapStatus.push_back(WiresSwapStatus::Local);
                localTgtsInC.push_back(vec);
            } else {
                size_t counts_global_wires = std::count_if(
                    statusWires.begin(),
                    statusWires.begin() + this->getNumLocalQubits(),
                    [](int i) { return i != WireStatus::Default; });
                size_t counts_local_wires_avail =
                    this->getNumLocalQubits() -
                    (vec.size() - counts_global_wires);

                if (counts_global_wires <= counts_local_wires_avail) {
                    tgtsSwapStatus.push_back(WiresSwapStatus::Swappable);

                    std::vector<int> localVec(vec.size());
                    std::transform(
                        vec.begin(), vec.end(), localVec.begin(),
                        [&](size_t x) { return static_cast<int>(x); });
                    auto wirePairs = createWirePairs(this->getNumLocalQubits(),
                                                     this->getTotalNumQubits(),
                                                     localVec, statusWires);
                    std::vector<size_t> localVecSizeT(localVec.size());
                    std::transform(
                        localVec.begin(), localVec.end(), localVecSizeT.begin(),
                        [&](int x) { return static_cast<size_t>(x); });
                    localTgtsInC.push_back(localVecSizeT);
                    tgtswirePairs.push_back(wirePairs);

                } else {
                    tgtsSwapStatus.push_back(WiresSwapStatus::UnSwappable);
                    localTgtsInC.push_back(vec);
                }
            }
        }
        auto threshold = WiresSwapStatus::Swappable;
        bool allLocal = std::all_of(
            tgtsSwapStatus.begin(), tgtsSwapStatus.end(),
            [&threshold](size_t status) { return status < threshold; });
        
        mpi_manager_.Barrier();

        if (allLocal) {
            expvalOnPauliBasis(pauli_words, tgts, expect_local);
        } else {
            size_t wirePairsIdx = 0;
            for (size_t i = 0; i < pauli_words.size(); i++) {
                if (tgtsSwapStatus[i] == WiresSwapStatus::Local) {
                    std::vector<std::string> pauli_words_idx(
                        1, std::string(pauli_words[i]));
                    std::vector<std::vector<size_t>> tgts_idx;
                    tgts_idx.push_back(localTgtsInC[i]);
                    std::vector<double> expval_local(1);

                    expvalOnPauliBasis(pauli_words_idx, tgts_idx, expval_local);
                    expect_local[i] = expval_local[0];
                } else if (tgtsSwapStatus[i] == WiresSwapStatus::Swappable) {
                    std::vector<std::string> pauli_words_idx(
                        1, std::string(pauli_words[i]));
                    std::vector<std::vector<size_t>> tgts_idx;
                    tgts_idx.push_back(localTgtsInC[i]);
                    std::vector<double> expval_local(1);

                    applyMPI_Dispatcher(tgtswirePairs[wirePairsIdx],
                                        &StateVectorCudaMPI::expvalOnPauliBasis,
                                        pauli_words_idx, tgts_idx,
                                        expval_local);
                    wirePairsIdx++;
                    expect_local[i] = expval_local[0];
                } else {
                
                    auto opsNames = pauliStringToOpNames(pauli_words[i]);
                    StateVectorCudaMPI<Precision> tmp(
                        this->getDataBuffer().getDevTag(),
                        this->getNumGlobalQubits(), this->getNumLocalQubits(),
                        this->getData());

                    for (size_t opsIdx = 0; opsIdx < tgts[i].size(); opsIdx++) {
                        std::vector<size_t> wires = {tgts[i][opsIdx]};
                        tmp.applyOperation({opsNames[opsIdx]},
                                           {tgts[i][opsIdx]}, {false});
                    }

                    expect_local[i] =
                        innerProdC_CUDA(
                            tmp.getData(), BaseType::getData(),
                            BaseType::getLength(),
                            BaseType::getDataBuffer().getDevTag().getDeviceID(),
                            BaseType::getDataBuffer().getDevTag().getStreamID(),
                            this->getCublasCaller())
                            .x;
                    PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
                    mpi_manager_.Barrier();
                }
            }
        }

        auto expect = mpi_manager_.allreduce<double>(expect_local, "sum");
        std::complex<Precision> result{0, 0};

        if constexpr (std::is_same_v<Precision, double>) {
            for (std::size_t idx = 0; idx < expect.size(); idx++) {
                result += expect[idx] * coeffs[idx];
            }
            return std::real(result);
        } else {
            std::vector<Precision> expect_cast(expect.size());
            std::transform(expect.begin(), expect.end(), expect_cast.begin(),
                           [](double x) { return static_cast<float>(x); });

            for (std::size_t idx = 0; idx < expect_cast.size(); idx++) {
                result += expect_cast[idx] * coeffs[idx];
            }

            return std::real(result);
        }
    }

    /**
     * @brief Utility method for samples.
     *
     * @param num_samples Number of Samples
     *
     * @return std::vector<size_t> A 1-d array storing the samples.
     * Each sample has a length equal to the number of qubits. Each sample can
     * be accessed using the stride sample_id*num_qubits, where sample_id is a
     * number between 0 and num_samples-1.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {
        double epsilon = 1e-15;
        size_t nSubSvs = 1UL << (this->getNumGlobalQubits());
        std::vector<double> rand_nums(num_samples);
        std::vector<size_t> samples(num_samples * this->getTotalNumQubits(), 0);

        size_t bitStringLen =
            this->getNumGlobalQubits() + this->getNumLocalQubits();

        std::vector<int> bitOrdering(bitStringLen);

        for (size_t i = 0; i < bitOrdering.size(); i++) {
            bitOrdering[i] = i;
        }

        std::vector<custatevecIndex_t> localBitStrings(num_samples);
        std::vector<custatevecIndex_t> globalBitStrings(num_samples);

        if (mpi_manager_.getRank() == 0) {
            for (size_t n = 0; n < num_samples; n++) {
                rand_nums[n] = (n + 1.0) / (num_samples + 2.0);
            }
        }

        mpi_manager_.Bcast<double>(rand_nums, 0);

        cudaDataType_t data_type;
        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        custatevecSamplerDescriptor_t sampler;

        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerCreate(
            /* custatevecHandle_t */ handle_.get(),
            /* const void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ this->getNumLocalQubits(),
            /* custatevecSamplerDescriptor_t * */ &sampler,
            /* uint32_t */ num_samples,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerPreprocess(
            /* custatevecHandle_t */ handle_.get(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* void* */ extraWorkspace,
            /* const size_t */ extraWorkspaceSizeInBytes));

        double subNorm = 0;
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerGetSquaredNorm(
            /* custatevecHandle_t */ handle_.get(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* double * */ &subNorm));

        int source = (mpi_manager_.getRank() - 1 + mpi_manager_.getSize()) %
                     mpi_manager_.getSize();
        int dest = (mpi_manager_.getRank() + 1) % mpi_manager_.getSize();

        double cumulative = 0;
        mpi_manager_.Scan<double>(subNorm, cumulative, "sum");

        double norm = cumulative;
        mpi_manager_.Bcast<double>(norm, mpi_manager_.getSize() - 1);

        double precumulative;
        mpi_manager_.Sendrecv<double>(cumulative, dest, precumulative, source);
        if (mpi_manager_.getRank() == 0) {
            precumulative = 0;
        }
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

        // Ensure the 'custatevecSamplerApplySubSVOffset' function can be called
        // successfully without reducing accuracy.
        if (precumulative == norm) {
            precumulative = norm - epsilon;
        }
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerApplySubSVOffset(
            /* custatevecHandle_t */ handle_.get(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* int32_t */ static_cast<int>(mpi_manager_.getRank()),
            /* uint32_t */ nSubSvs,
            /* double */ precumulative,
            /* double */ norm));

        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        auto low = std::lower_bound(rand_nums.begin(), rand_nums.end(),
                                    cumulative / norm);
        int shotOffset = std::distance(rand_nums.begin(), low);
        if (mpi_manager_.getRank() == (mpi_manager_.getSize() - 1)) {
            shotOffset = num_samples;
        }

        int preshotOffset;
        mpi_manager_.Sendrecv<int>(shotOffset, dest, preshotOffset, source);
        if (mpi_manager_.getRank() == 0) {
            preshotOffset = 0;
        }

        int nSubShots = shotOffset - preshotOffset;
        if (nSubShots > 0) {
            PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerSample(
                /* custatevecHandle_t */ handle_.get(),
                /* custatevecSamplerDescriptor_t */ sampler,
                /* custatevecIndex_t* */ &localBitStrings[preshotOffset],
                /* const int32_t * */ bitOrdering.data(),
                /* const uint32_t */ bitStringLen,
                /* const double * */ &rand_nums[preshotOffset],
                /* const uint32_t */ nSubShots,
                /* enum custatevecSamplerOutput_t */
                CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER));
        }

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerDestroy(sampler));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        }

        mpi_manager_.Allreduce<custatevecIndex_t>(localBitStrings,
                                                  globalBitStrings, "sum");

        for (size_t i = 0; i < num_samples; i++) {
            for (size_t j = 0; j < bitStringLen; j++) {
                samples[i * bitStringLen + (bitStringLen - 1 - j)] =
                    (globalBitStrings[i] >> j) & 1U;
            }
        }
        return samples;
    }

  private:
    using ParFunc = std::function<void(const std::vector<size_t> &, bool,
                                       const std::vector<Precision> &)>;
    using FMap = std::unordered_map<std::string, ParFunc>;
    const FMap par_gates_{
        {"RX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRX(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        {"RY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRY(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        {"RZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRZ(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        {"PhaseShift",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyPhaseShift(std::forward<decltype(wires)>(wires),
                             std::forward<decltype(adjoint)>(adjoint),
                             std::forward<decltype(params[0])>(params[0]));
         }},
        {"MultiRZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyMultiRZ(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingXX(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingYY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingYY(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingZZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingZZ(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRX(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRY(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRZ(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitation",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationPlus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationMinus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitation",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationPlus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationMinus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"ControlledPhaseShift",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyControlledPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"Rot",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRot(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params)>(params));
         }},
        {"CRot", [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRot(std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
         }}};

    const std::unordered_map<std::string, custatevecPauli_t> native_gates_{
        {"RX", CUSTATEVEC_PAULI_X},       {"RY", CUSTATEVEC_PAULI_Y},
        {"RZ", CUSTATEVEC_PAULI_Z},       {"CRX", CUSTATEVEC_PAULI_X},
        {"CRY", CUSTATEVEC_PAULI_Y},      {"CRZ", CUSTATEVEC_PAULI_Z},
        {"Identity", CUSTATEVEC_PAULI_I}, {"I", CUSTATEVEC_PAULI_I}};

    /**
     * @brief Normalize the index ordering to match PennyLane.
     *
     * @tparam IndexType Integer value type.
     * @param indices Given indices to transform.
     */
    template <typename IndexType>
    inline auto NormalizeIndices(std::vector<IndexType> indices)
        -> std::vector<IndexType> {
        std::vector<IndexType> t_indices(std::move(indices));
        std::transform(t_indices.begin(), t_indices.end(), t_indices.begin(),
                       [&](IndexType i) -> IndexType {
                           return BaseType::getNumQubits() - 1 - i;
                       });
        return t_indices;
    }

    /**
     * @brief Get expectation value for a sum of Pauli words.
     *
     * @param pauli_words Vector of Pauli-words to evaluate expectation value.
     * @param tgts Coupled qubit index to apply each Pauli term.
     * @param coeffs Numpy array buffer of size |pauli_words|
     * @return auto Expectation value.
     */
    auto expvalOnPauliBasis(const std::vector<std::string> &pauli_words,
                            const std::vector<std::vector<std::size_t>> &tgts,
                            std::vector<double> &local_expect) {

        uint32_t nIndexBits = static_cast<uint32_t>(this->getNumLocalQubits());
        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        // Note: due to API design, cuStateVec assumes this is always a double.
        // Push NVIDIA to move this to behind API for future releases, and
        // support 32/64 bits.
        // std::vector<double> expect(pauli_words.size());

        std::vector<std::vector<custatevecPauli_t>> pauliOps;

        std::vector<custatevecPauli_t *> pauliOps_ptr;

        for (auto &p_word : pauli_words) {
            pauliOps.push_back(cuUtil::pauliStringToEnum(p_word));
            pauliOps_ptr.push_back((*pauliOps.rbegin()).data());
        }

        std::vector<std::vector<int32_t>> basisBits;
        std::vector<int32_t *> basisBits_ptr;
        std::vector<uint32_t> n_basisBits;

        for (auto &wires : tgts) {
            std::vector<int32_t> wiresInt(wires.size());
            std::transform(wires.begin(), wires.end(), wiresInt.begin(),
                           [&](std::size_t x) { return static_cast<int>(x); });
            basisBits.push_back(wiresInt);
            basisBits_ptr.push_back((*basisBits.rbegin()).data());
            n_basisBits.push_back(wiresInt.size());
        }

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationsOnPauliBasis(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double* */ local_expect.data(),
            /* const custatevecPauli_t ** */
            const_cast<const custatevecPauli_t **>(pauliOps_ptr.data()),
            /* const uint32_t */ static_cast<uint32_t>(pauliOps.size()),
            /* const int32_t ** */
            const_cast<const int32_t **>(basisBits_ptr.data()),
            /* const uint32_t */ n_basisBits.data()));
    }

    /**
     * @brief Apply parametric Pauli gates to local statevector using custateVec
     * calls.
     *
     * @param pauli_words List of Pauli words representing operation.
     * @param ctrls Control wires
     * @param tgts target wires.
     * @param param Gate parameter.
     * @param use_adjoint Take adjoint of operation.
     */
    void applyCuSVPauliGate(const std::vector<std::string> &pauli_words,
                            std::vector<int> &ctrls, std::vector<int> &tgts,
                            Precision param, bool use_adjoint = false) {
        int nIndexBits = BaseType::getNumQubits();

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::vector<custatevecPauli_t> pauli_enums;
        pauli_enums.reserve(pauli_words.size());
        for (const auto &pauli_str : pauli_words) {
            pauli_enums.push_back(native_gates_.at(pauli_str));
        }
        const auto local_angle = (use_adjoint) ? param / 2 : -param / 2;

        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyPauliRotation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double */ local_angle,
            /* const custatevecPauli_t* */ pauli_enums.data(),
            /* const int32_t* */ tgts.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrls.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size()));
    }

    /**
     * @brief Apply parametric Pauli gates using custateVec calls.
     *
     * @param pauli_words List of Pauli words representing operation.
     * @param ctrls Control wires
     * @param tgts target wires.
     * @param param Parametric gate parameter.
     * @param use_adjoint Take adjoint of operation.
     */
    void applyParametricPauliGate(const std::vector<std::string> &pauli_words,
                                  std::vector<std::size_t> ctrls,
                                  std::vector<std::size_t> tgts,
                                  Precision param, bool use_adjoint = false) {
        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // Initialize a vector to store the status of wires and default its
        // elements as zeros, which assumes there is no target and control wire.
        std::vector<int> statusWires(this->getTotalNumQubits(),
                                     WireStatus::Default);

        // Update wire status based on the gate information
        for (size_t i = 0; i < ctrlsInt.size(); i++) {
            statusWires[ctrlsInt[i]] = WireStatus::Control;
        }
        // Update wire status based on the gate information
        for (size_t i = 0; i < tgtsInt.size(); i++) {
            statusWires[tgtsInt[i]] = WireStatus::Target;
        }

        int StatusGlobalWires = std::reduce(
            statusWires.begin() + this->getNumLocalQubits(), statusWires.end());

        mpi_manager_.Barrier();

        if (!StatusGlobalWires) {
            applyCuSVPauliGate(pauli_words, ctrlsInt, tgtsInt, param,
                               use_adjoint);
        } else {
            size_t counts_global_wires =
                std::count_if(statusWires.begin(),
                              statusWires.begin() + this->getNumLocalQubits(),
                              [](int i) { return i != WireStatus::Default; });
            size_t counts_local_wires =
                ctrlsInt.size() + tgtsInt.size() - counts_global_wires;
            PL_ABORT_IF(
                counts_global_wires >
                    (this->getNumLocalQubits() - counts_local_wires),
                "There is not enough local wires for bit swap operation.");

            std::vector<int> localCtrls(ctrlsInt);
            std::vector<int> localTgts(tgtsInt);

            auto wirePairs = createWirePairs(
                this->getNumLocalQubits(), this->getTotalNumQubits(),
                localCtrls, localTgts, statusWires);

            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

            applyMPI_Dispatcher(
                wirePairs, &StateVectorCudaMPI::applyCuSVPauliGate, pauli_words,
                localCtrls, localTgts, param, use_adjoint);
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        }
    }

    /**
     * @brief Apply a given host or device-stored array representing the gate
     * `matrix` to the local statevector at qubit indices given by `tgts` and
     * control-lines given by `ctrls`. The adjoint can be taken by setting
     * `use_adjoint` to true.
     *
     * @param matrix Host- or device data array in row-major order representing
     * a given gate.
     * @param ctrls Control line qubits.
     * @param tgts Target qubits.
     * @param use_adjoint Use adjoint of given gate.
     */
    void applyCuSVDeviceMatrixGate(const CFP_t *matrix,
                                   const std::vector<int> &ctrls,
                                   const std::vector<int> &tgts,
                                   bool use_adjoint = false) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        int nIndexBits = BaseType::getNumQubits();

        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrixGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const uint32_t */ tgts.size(),
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        // allocate external workspace if necessary
        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        // apply gate
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrix(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const int32_t* */ tgts.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrls.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
    }

    /**
     * @brief Apply a given host or device-stored array representing the gate
     * `matrix` to the statevector at qubit indices given by `tgts` and
     * control-lines given by `ctrls`. The adjoint can be taken by setting
     * `use_adjoint` to true.
     *
     * @param matrix Host- or device data array in row-major order representing
     * a given gate.
     * @param ctrls Control line qubits.
     * @param tgts Target qubits.
     * @param use_adjoint Use adjoint of given gate.
     */
    void applyDeviceMatrixGate(const CFP_t *matrix,
                               const std::vector<std::size_t> &ctrls,
                               const std::vector<std::size_t> &tgts,
                               bool use_adjoint = false) {

        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // Initialize a vector to store the status of wires and default its
        // elements as zeros, which assumes there is no target and control wire.
        std::vector<int> statusWires(this->getTotalNumQubits(),
                                     WireStatus::Default);

        // Update wire status based on the gate information
        for (size_t i = 0; i < ctrlsInt.size(); i++) {
            statusWires[ctrlsInt[i]] = WireStatus::Control;
        }
        // Update wire status based on the gate information
        for (size_t i = 0; i < tgtsInt.size(); i++) {
            statusWires[tgtsInt[i]] = WireStatus::Target;
        }

        int StatusGlobalWires = std::reduce(
            statusWires.begin() + this->getNumLocalQubits(), statusWires.end());

        mpi_manager_.Barrier();

        if (!StatusGlobalWires) {
            applyCuSVDeviceMatrixGate(matrix, ctrlsInt, tgtsInt, use_adjoint);
        } else {
            size_t counts_global_wires =
                std::count_if(statusWires.begin(),
                              statusWires.begin() + this->getNumLocalQubits(),
                              [](int i) { return i != WireStatus::Default; });
            size_t counts_local_wires =
                ctrlsInt.size() + tgtsInt.size() - counts_global_wires;
            PL_ABORT_IF(
                counts_global_wires >
                    (this->getNumLocalQubits() - counts_local_wires),
                "There is not enough local wires for bit swap operation.");

            std::vector<int> localCtrls = ctrlsInt;
            std::vector<int> localTgts = tgtsInt;

            auto wirePairs = createWirePairs(
                this->getNumLocalQubits(), this->getTotalNumQubits(),
                localCtrls, localTgts, statusWires);

            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

            applyMPI_Dispatcher(wirePairs,
                                &StateVectorCudaMPI::applyCuSVDeviceMatrixGate,
                                matrix, localCtrls, localTgts, use_adjoint);
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        }
    }

    /**
     * @brief Apply a given host-matrix `matrix` to the local state vector at
     * qubit indices given by `tgts` and control-lines given by `ctrls`. The
     * adjoint can be taken by setting `use_adjoint` to true.
     *
     * @param matrix Host-data vector in row-major order of a given gate.
     * @param ctrls Control line qubits.
     * @param tgts Target qubits.
     * @param use_adjoint Use adjoint of given gate.
     */
    void applyCuSVHostMatrixGate(const std::vector<CFP_t> &matrix,
                                 const std::vector<int> &ctrls,
                                 const std::vector<int> &tgts,
                                 bool use_adjoint = false) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        int nIndexBits = BaseType::getNumQubits();

        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrixGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const uint32_t */ tgts.size(),
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        // allocate external workspace if necessary
        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        // apply gate
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrix(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const int32_t* */ tgts.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrls.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
    }

    /**
     * @brief Apply a given host-matrix `matrix` to the state vector at qubit
     * indices given by `tgts` and control-lines given by `ctrls`. The adjoint
     * can be taken by setting `use_adjoint` to true.
     *
     * @param matrix Host-data vector in row-major order of a given gate.
     * @param ctrls Control line qubits.
     * @param tgts Target qubits.
     * @param use_adjoint Use adjoint of given gate.
     */
    void applyHostMatrixGate(const std::vector<CFP_t> &matrix,
                             const std::vector<std::size_t> &ctrls,
                             const std::vector<std::size_t> &tgts,
                             bool use_adjoint = false) {
        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // Initialize a vector to store the status of wires and default its
        // elements as zeros, which assumes there is no target and control wire.
        std::vector<int> statusWires(this->getTotalNumQubits(),
                                     WireStatus::Default);

        // Update wire status based on the gate information
        for (size_t i = 0; i < ctrlsInt.size(); i++) {
            statusWires[ctrlsInt[i]] = WireStatus::Control;
        }
        // Update wire status based on the gate information
        for (size_t i = 0; i < tgtsInt.size(); i++) {
            statusWires[tgtsInt[i]] = WireStatus::Target;
        }

        int StatusGlobalWires = std::reduce(
            statusWires.begin() + this->getNumLocalQubits(), statusWires.end());

        mpi_manager_.Barrier();

        if (!StatusGlobalWires) {
            applyCuSVHostMatrixGate(matrix, ctrlsInt, tgtsInt, use_adjoint);
        } else {
            size_t counts_global_wires =
                std::count_if(statusWires.begin(),
                              statusWires.begin() + this->getNumLocalQubits(),
                              [](int i) { return i != WireStatus::Default; });
            size_t counts_local_wires =
                ctrlsInt.size() + tgtsInt.size() - counts_global_wires;

            PL_ABORT_IF(
                counts_global_wires >
                    (this->getNumLocalQubits() - counts_local_wires),
                "There is not enough local wires for bit swap operation.");

            std::vector<int> localCtrls = ctrlsInt;
            std::vector<int> localTgts = tgtsInt;

            auto wirePairs = createWirePairs(
                this->getNumLocalQubits(), this->getTotalNumQubits(),
                localCtrls, localTgts, statusWires);

            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

            applyMPI_Dispatcher(wirePairs,
                                &StateVectorCudaMPI::applyCuSVHostMatrixGate,
                                matrix, localCtrls, localTgts, use_adjoint);
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        }
    }

    void applyHostMatrixGate(const std::vector<std::complex<Precision>> &matrix,
                             const std::vector<std::size_t> &ctrls,
                             const std::vector<std::size_t> &tgts,
                             bool use_adjoint = false) {
        std::vector<CFP_t> matrix_cu(matrix.size());
        for (std::size_t i = 0; i < matrix.size(); i++) {
            matrix_cu[i] =
                cuUtil::complexToCu<std::complex<Precision>>(matrix[i]);
        }

        applyHostMatrixGate(matrix_cu, ctrls, tgts, use_adjoint);
    }

    /**
     * @brief Get expectation of a given host-defined matrix.
     *
     * @param matrix Host-defined row-major order gate matrix.
     * @param tgts Target qubits.
     * @param expect Local expectation value.
     * @return auto Expectation value.
     */
    void getCuSVExpectationValueHostMatrix(const std::vector<CFP_t> &matrix,
                                           const std::vector<int> &tgts,
                                           CFP_t &expect) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        size_t nIndexBits = BaseType::getNumQubits();
        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect,
            /* cudaDataType_t */ data_type,
            /* double* */ nullptr,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgts.data(),
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes) {
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        }
    }

    auto getExpectationValueHostMatrix(const std::vector<CFP_t> &matrix,
                                       const std::vector<std::size_t> &tgts) {

        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // Initialize a vector to store the status of wires and default its
        // elements as zeros, which assumes there is no target and control wire.
        std::vector<int> statusWires(this->getTotalNumQubits(),
                                     WireStatus::Default);

        // Update wire status based on the gate information
        for (size_t i = 0; i < tgtsInt.size(); i++) {
            statusWires[tgtsInt[i]] = WireStatus::Target;
        }

        int StatusGlobalWires = std::reduce(
            statusWires.begin() + this->getNumLocalQubits(), statusWires.end());

        mpi_manager_.Barrier();

        CFP_t local_expect;
        if (!StatusGlobalWires) {
            getCuSVExpectationValueHostMatrix(matrix, tgtsInt, local_expect);
        } else {
            std::vector<int> localTgts = tgtsInt;

            auto wirePairs = createWirePairs(this->getNumLocalQubits(),
                                             this->getTotalNumQubits(),
                                             localTgts, statusWires);

            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

            applyMPI_Dispatcher(
                wirePairs,
                &StateVectorCudaMPI::getCuSVExpectationValueHostMatrix, matrix,
                localTgts, local_expect);
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        }
        auto expect = mpi_manager_.allreduce<CFP_t>(local_expect, "sum");
        return expect;
    }

    /**
     * @brief Get expectation of a given host or device defined array.
     *
     * @param matrix Host or device defined row-major order gate matrix array.
     * @param tgts Target qubits.
     * @param expect Local expectation value.
     * @return auto Expectation value.
     */
    void getCuSVExpectationValueDeviceMatrix(const CFP_t *matrix,
                                             const std::vector<int> &tgts,
                                             CFP_t &expect) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        size_t nIndexBits = BaseType::getNumQubits();
        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect,
            /* cudaDataType_t */ data_type,
            /* double* */ nullptr,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgts.data(),
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes) {
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        }
    }

    auto getExpectationValueDeviceMatrix(const CFP_t *matrix,
                                         const std::vector<std::size_t> &tgts) {
        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // Initialize a vector to store the status of wires and default its
        // elements as zeros, which assumes there is no target and control wire.
        std::vector<int> statusWires(this->getTotalNumQubits(),
                                     WireStatus::Default);

        // Update wire status based on the gate information
        for (size_t i = 0; i < tgtsInt.size(); i++) {
            statusWires[tgtsInt[i]] = WireStatus::Target;
        }

        int StatusGlobalWires = std::reduce(
            statusWires.begin() + this->getNumLocalQubits(), statusWires.end());

        mpi_manager_.Barrier();

        CFP_t local_expect;
        if (!StatusGlobalWires) {
            getCuSVExpectationValueDeviceMatrix(matrix, tgtsInt, local_expect);
        } else {
            std::vector<int> localTgts = tgtsInt;

            auto wirePairs = createWirePairs(this->getNumLocalQubits(),
                                             this->getTotalNumQubits(),
                                             localTgts, statusWires);

            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

            applyMPI_Dispatcher(
                wirePairs,
                &StateVectorCudaMPI::getCuSVExpectationValueDeviceMatrix,
                matrix, localTgts, local_expect);
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        }
        auto expect = mpi_manager_.allreduce<CFP_t>(local_expect, "sum");
        return expect;
    }

    /**
     * @brief MPI dispatcher for the target and control gates at global qubits.
     *
     * @tparam F Return type of the callable.
     * @tparam Args Types of arguments of t the callable.
     *
     * @param wirePairs Vector of wire pairs for bit index swap operations.
     * @param functor The callable.
     * @param args Arguments of the callable.
     */
    template <typename F, typename... Args>
    void applyMPI_Dispatcher(std::vector<int2> &wirePairs, F &&functor,
                             Args &&...args) {
        int maskBitString[] = {}; // specify the values of mask qubits
        int maskOrdering[] = {};  // specify the mask qubits

        cudaDataType_t svDataType;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            svDataType = CUDA_C_64F;
        } else {
            svDataType = CUDA_C_32F;
        }
        //
        // create distributed index bit swap scheduler
        //
        custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler;
        PL_CUSTATEVEC_IS_SUCCESS(custatevecDistIndexBitSwapSchedulerCreate(
            /* custatevecHandle_t */ handle_.get(),
            /* custatevecDistIndexBitSwapSchedulerDescriptor_t */
            &scheduler,
            /* uint32_t */ this->getNumGlobalQubits(),
            /* uint32_t */ this->getNumLocalQubits()));

        // set the index bit swaps to the scheduler
        // nSwapBatches is obtained by the call.  This value specifies the
        // number of loops
        unsigned nSwapBatches = 0;
        PL_CUSTATEVEC_IS_SUCCESS(
            custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(
                /* custatevecHandle_t */ handle_.get(),
                /* custatevecDistIndexBitSwapSchedulerDescriptor_t */
                scheduler,
                /* const int2* */ wirePairs.data(),
                /* const uint32_t */
                static_cast<unsigned>(wirePairs.size()),
                /* const int32_t* */ maskBitString,
                /* const int32_t* */ maskOrdering,
                /* const uint32_t */ 0,
                /* uint32_t* */ &nSwapBatches));

        //
        // the main loop of index bit swaps
        //
        constexpr size_t nLoops = 2;
        for (size_t loop = 0; loop < nLoops; ++loop) {
            for (int swapBatchIndex = 0;
                 swapBatchIndex < static_cast<int>(nSwapBatches);
                 ++swapBatchIndex) {
                // get parameters
                custatevecSVSwapParameters_t parameters;
                PL_CUSTATEVEC_IS_SUCCESS(
                    custatevecDistIndexBitSwapSchedulerGetParameters(
                        /* custatevecHandle_t */ handle_.get(),
                        /* custatevecDistIndexBitSwapSchedulerDescriptor_t*/
                        scheduler,
                        /* const int32_t */ swapBatchIndex,
                        /* const int32_t */ mpi_manager_.getRank(),
                        /* custatevecSVSwapParameters_t* */
                        &parameters));

                // the rank of the communication endpoint is
                // parameters.dstSubSVIndex as "rank == subSVIndex" is assumed
                // in the present sample.
                int rank = parameters.dstSubSVIndex;
                // set parameters to the worker
                PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerSetParameters(
                    /* custatevecHandle_t */ handle_.get(),
                    /* custatevecSVSwapWorkerDescriptor_t */
                    this->getSwapWorker(),
                    /* const custatevecSVSwapParameters_t* */
                    &parameters,
                    /* int */ rank));
                PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
                PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize())
                // execute swap
                PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerExecute(
                    /* custatevecHandle_t */ handle_.get(),
                    /* custatevecSVSwapWorkerDescriptor_t */
                    this->getSwapWorker(),
                    /* custatevecIndex_t */ 0,
                    /* custatevecIndex_t */ parameters.transferSize));
                // all internal CUDA calls are serialized on localStream
                PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize())
                mpi_manager_.Barrier();
            }
            if (loop == 0) {
                std::invoke(std::forward<F>(functor), this,
                            std::forward<Args>(args)...);
            }
            // synchronize all operations on device
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
            mpi_manager_.Barrier();
        }

        PL_CUSTATEVEC_IS_SUCCESS(custatevecDistIndexBitSwapSchedulerDestroy(
            handle_.get(), scheduler));
    }
};

}; // namespace Pennylane