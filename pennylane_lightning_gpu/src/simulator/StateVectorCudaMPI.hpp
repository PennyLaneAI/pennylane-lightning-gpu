// Copyright 2022 Xanadu Quantum Technologies Inc.

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
 * @file StateVectorCudaBase.hpp
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
#include <mpi.h>

#include "Constant.hpp"
#include "Error.hpp"
#include "StateVectorCudaBase.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"
#include "mpiWorker.hpp"
#include "mpi_helpers.hpp"

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

    int numGlobalQubits_;
    int rank_;
    int numProcs_;
    int deviceId_;

    MPI_Comm mpiCommunicator_;
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
    StateVectorCudaMPI(MPI_Comm MPICommunicator, size_t num_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_qubits),
          mpiCommunicator_(MPICommunicator), handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker(
              handle_.get(), MPICommunicator, BaseType::getData(), num_qubits,
              localStream_.get())),
          gate_cache_(true) {
        MPI_Comm_size(mpiCommunicator_, &numProcs_);
        MPI_Comm_rank(mpiCommunicator_, &rank_);
        numGlobalQubits_ = 0;
        while ((1 << numGlobalQubits_) < numProcs_) {
            ++numGlobalQubits_;
        }
        int nDevices = 0;
        PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
        deviceId_ = rank_ % nDevices;
    };

    ~StateVectorCudaMPI() {}

    void initSV_MPI(bool async = false) {
        size_t index = 0;
        const std::complex<Precision> value = {1, 0};
        BaseType::getData()->zeroInit();
        setBasisState(value, index, false);
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
        int rankId = index >> BaseType::getNumQubits();
        int local_index = (rankId << BaseType::getNumQubits()) ^ index;
        BaseType::getDataBuffer().zeroInit();

        CFP_t value_cu = cuUtil::complexToCu<std::complex<Precision>>(value);
        auto stream_id = localStream_.get();

        if (rank_ == rankId) {
            setBasisState_CUDA(BaseType::getData(), value_cu, local_index,
                               async, stream_id);
        }

        MPI_Barrier(mpiCommunicator_);
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

        for (int i = 0; i < num_indices; i++) {
            int index = indices[i];
            int rankId = index >> BaseType::getNumQubits();
            if (rankId == rank_) {
                int local_index = index ^ (rankId << BaseType::getNumQubits());
                indices_local.push_back(local_index);
                values_local.push_back(values[i]);
            }
        }

        auto device_id = BaseType::getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = BaseType::getDataBuffer().getDevTag().getStreamID();

        MPI_Barrier(mpiCommunicator_);

        index_type num_elements = indices_local.size();

        DataBuffer<index_type, int> d_indices{
            static_cast<std::size_t>(num_elements), device_id, stream_id, true};

        DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(num_elements),
                                        device_id, stream_id, true};

        d_indices.CopyHostDataToGpu(indices_local.data(), d_indices.getLength(),
                                    async);
        d_values.CopyHostDataToGpu(values_local.data(), d_values.getLength(),
                                   async);
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        MPI_Barrier(mpiCommunicator_);

        setStateVector_CUDA(BaseType::getData(), num_elements,
                            d_values.getData(), d_indices.getData(),
                            thread_per_block, stream_id);

        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        MPI_Barrier(mpiCommunicator_);
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
     * @brief STL-fiendly variant of `applyOperation(
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
     * @param params
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
        std::vector<double> probabilities(Util::exp2(wires.size()));
        // this should be built upon by the wires not participating
        int maskLen =
            0; // static_cast<int>(BaseType::getNumQubits() - wires.size());
        int *maskBitString = nullptr; //
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
        wires_local.reserve(wires.size());
        std::vector<int> wires_global;
        wires_global.reserve(wires.size());

        // Partition the vector into two vectors
        auto it =
            std::partition(wires_int.begin(), wires_int.end(), [&](int i) {
                return i < (this->getNumLocalQubits());
            });
        wires_local.assign(wires_int.begin(), it);
        wires_global.assign(it, wires_int.end());

        // no need to create new MPI communicator group if wires_global.size()
        // == 0
        if (wires_global.size() ==
            0 /*&& wires_local.size() == wires.size()*/) {
            std::vector<double> local_probabilities(
                Util::exp2(wires_local.size()));
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
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
            PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));
            PL_MPI_IS_SUCCESS(
                MPI_Allreduce(local_probabilities.data(), probabilities.data(),
                              local_probabilities.size(), MPI_DOUBLE, MPI_SUM,
                              mpiCommunicator_));
            return probabilities;

        } else {
            std::vector<double> local_probabilities(
                Util::exp2(wires_local.size()));
            std::vector<double> subgroup_probabilities(
                Util::exp2(wires_local.size()));
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
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
            PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));
            // create new MPI communicator groups
            int subCommGroupId = 0;
            std::vector<int> bitArray(wires_global.size());
            for (int i = 0; i < static_cast<int>(wires_global.size()); i++) {
                int mask = 1 << (wires_global[i] - this->getNumLocalQubits());
                int bitValue = this->getCommRank() & mask;
                subCommGroupId +=
                    bitValue << (wires_global[i] - this->getNumLocalQubits());
            }
            MPI_Comm subComm0;
            PL_MPI_IS_SUCCESS(MPI_Comm_split(mpiCommunicator_, subCommGroupId,
                                             this->getCommRank(), &subComm0));
            int subRank0, subSize0;
            PL_MPI_IS_SUCCESS(MPI_Comm_rank(subComm0, &subRank0));
            PL_MPI_IS_SUCCESS(MPI_Comm_size(subComm0, &subSize0));
            // PL_MPI_IS_SUCCESS(MPI_Reduce(
            PL_MPI_IS_SUCCESS(MPI_Allreduce(
                local_probabilities.data(), subgroup_probabilities.data(),
                local_probabilities.size(), MPI_DOUBLE, MPI_SUM, subComm0));
            //    local_probabilities.size(), MPI_DOUBLE, MPI_SUM, 0,
            //    subComm0));

            PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

            MPI_Comm subComm1;
            PL_MPI_IS_SUCCESS(MPI_Comm_split(mpiCommunicator_, subRank0,
                                             this->getCommRank(), &subComm1));
            int subRank1, subSize1;
            PL_MPI_IS_SUCCESS(MPI_Comm_rank(subComm1, &subRank1));
            PL_MPI_IS_SUCCESS(MPI_Comm_size(subComm1, &subSize1));

            // if (subRank0 == 0) {
            // PL_MPI_IS_SUCCESS(MPI_Gather(subgroup_probabilities.data(),
            PL_MPI_IS_SUCCESS(MPI_Allgather(
                subgroup_probabilities.data(), subgroup_probabilities.size(),
                MPI_DOUBLE, probabilities.data(), subgroup_probabilities.size(),
                MPI_DOUBLE, subComm1));
            //                             MPI_DOUBLE, 0, subComm1));
            //}

            /*
            if(subRank1==0 && subRank0 == 0){
                //int root = this->getCommRank();
                PL_MPI_IS_SUCCESS(MPI_Bcast(probabilities.data(),
                                            probabilities.size(), MPI_DOUBLE,
                                            0, mpiCommunicator_));
            }
            */

            PL_MPI_IS_SUCCESS(MPI_Comm_free(&subComm0));
            PL_MPI_IS_SUCCESS(MPI_Comm_free(&subComm1));

            PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));
            return probabilities;
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

        size_t nSubSvs = 1 << (this->getNumGlobalQubits());
        std::vector<double> rand_nums(num_samples);
        std::vector<size_t> samples(num_samples * this->getTotalNumQubits(), 0);

        size_t bitStringLen =
            this->getNumGlobalQubits() + this->getNumLocalQubits();

        std::vector<int> bitOrdering(bitStringLen);
        for (std::size_t i = 0; i < bitOrdering.size(); i++) {
            bitOrdering[i] = i;
        }

        std::vector<custatevecIndex_t> localBitStrings(num_samples);
        std::vector<custatevecIndex_t> globalBitStrings(num_samples);

        if (this->getCommRank() == 0) {
            for (size_t n = 0; n < num_samples; n++) {
                rand_nums[n] = (n + 1.0) / (num_samples + 2.0);
            }
        }

        PL_MPI_IS_SUCCESS(MPI_Bcast(rand_nums.data(), rand_nums.size(),
                                    MPI_DOUBLE, 0, mpiCommunicator_));

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
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerPreprocess(
            /* custatevecHandle_t */ handle_.get(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* void* */ extraWorkspace,
            /* const size_t */ extraWorkspaceSizeInBytes));
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        double subNorm = 0;
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerGetSquaredNorm(
            /* custatevecHandle_t */ handle_.get(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* double * */ &subNorm));
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        // double cumulative=0;
        // PL_MPI_IS_SUCCESS(MPI_Scan(&subNorm, &cumulative, 1, MPI_DOUBLE,
        //                            MPI_SUM, mpiCommunicator_));
        // PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        std::vector<double> cumulativeArray(this->getCommSize());
        MPI_Allgather(&subNorm, 1, MPI_DOUBLE, cumulativeArray.data(), 1,
                      MPI_DOUBLE, mpiCommunicator_);

        std::partial_sum(cumulativeArray.begin(), cumulativeArray.end(),
                         cumulativeArray.begin());

        double norm = cumulativeArray.back();

        // double norm = cumulative;

        // PL_MPI_IS_SUCCESS(MPI_Bcast(
        //     &norm, 1, MPI_DOUBLE, this->getCommSize() - 1,
        //     MPI_COMM_WORLD)); // get last cumulative for all processes
        // double norm = 1;

        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        /*
        double precumulative;
        MPI_Status status;
        PL_MPI_IS_SUCCESS(
            MPI_Sendrecv(&cumulative, 1, MPI_DOUBLE,
                         (this->getCommRank() + 1) % this->getCommSize(), 0,
                         &precumulative, 1, MPI_DOUBLE,
                         (this->getCommRank() - 1 + this->getCommSize()) %
                             this->getCommSize(),
                         1, mpiCommunicator_, &status));

        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));
        */
        double precumulative;
        if (this->getCommRank() == 0) {
            precumulative = 0;
        } else {
            precumulative = cumulativeArray[this->getCommRank() - 1];
        }

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerApplySubSVOffset(
            /* custatevecHandle_t */ handle_.get(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* int32_t */ this->getCommRank(),
            /* uint32_t */ nSubSvs,
            /* double */ precumulative,
            /* double */ norm));
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        int shotOffset = 0;
        auto low =
            std::lower_bound(rand_nums.begin(), rand_nums.end(),
                             cumulativeArray[this->getCommRank()] / norm);
        int pos = std::distance(rand_nums.begin(), low);

        shotOffset = pos;

        if (this->getCommRank() == (this->getCommSize() - 1)) {
            shotOffset = num_samples;
        }

        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        MPI_Status status;
        int nSubShots;
        int preshotOffset;
        PL_MPI_IS_SUCCESS(
            MPI_Sendrecv(&shotOffset, 1, MPI_INT,
                         (this->getCommRank() + 1) % this->getCommSize(), 0,
                         &preshotOffset, 1, MPI_DOUBLE,
                         (this->getCommRank() - 1 + this->getCommSize()) %
                             this->getCommSize(),
                         0, mpiCommunicator_, &status));

        if (this->getCommRank() == 0) {
            preshotOffset = 0;
        }

        nSubShots = shotOffset - preshotOffset;

        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

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
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerDestroy(sampler));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        }

        PL_MPI_IS_SUCCESS(MPI_Allreduce(
            localBitStrings.data(), globalBitStrings.data(),
            globalBitStrings.size(), MPI_INT64_T, MPI_SUM, mpiCommunicator_));

        for (size_t i = 0; i < num_samples; i++) {
            for (size_t j = 0; j < bitStringLen; j++) {
                samples[i * bitStringLen + (bitStringLen - 1 - j)] =
                    (globalBitStrings[i] >> j) & 1U;
            }
        }

        return samples;
    }

    auto getCommRank() -> int { return rank_; }

    auto getCommSize() -> int { return numProcs_; };

    auto getTotalNumQubits() -> int {
        return numGlobalQubits_ + BaseType::getNumQubits();
    }

    auto getNumGlobalQubits() -> int { return numGlobalQubits_; }

    auto getNumLocalQubits() -> int { return BaseType::getNumQubits(); }

    auto getSwapWorker() -> custatevecSVSwapWorkerDescriptor_t {
        return svSegSwapWorker_.get();
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
     * @param angle Rotation angle.
     * @param pauli_words List of Pauli words representing operation.
     * @param ctrls Control wires
     * @param tgts target wires.
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

        std::vector<int> totalWires(this->getTotalNumQubits(), 0);

        bool BSwapReq = IsIndexSwapRequired(this->getNumLocalQubits(),
                                            this->getTotalNumQubits(), ctrlsInt,
                                            tgtsInt, totalWires);

        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        if (!BSwapReq) {
            applyCuSVPauliGate(pauli_words, ctrlsInt, tgtsInt, param,
                               use_adjoint);
        } else {
            std::vector<int> localCtrls(ctrlsInt);
            std::vector<int> localTgts(tgtsInt);
            auto wirePairs = createOperationWires(
                this->getNumLocalQubits(), this->getTotalNumQubits(),
                localCtrls, localTgts, totalWires);
            applyGate_MPI(wirePairs, &StateVectorCudaMPI::applyCuSVPauliGate,
                          pauli_words, localCtrls, localTgts, param,
                          use_adjoint);
        }
    }

    void applyCuSVMatrixGate(const CFP_t *matrix, const std::vector<int> &ctrls,
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

        std::vector<int> totalWires(this->getTotalNumQubits(), 0);

        bool BSwapReq = IsIndexSwapRequired(this->getNumLocalQubits(),
                                            this->getTotalNumQubits(), ctrlsInt,
                                            tgtsInt, totalWires);

        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        if (!BSwapReq) {
            applyCuSVMatrixGate(matrix, ctrlsInt, tgtsInt, use_adjoint);
        } else {
            std::vector<int> localCtrls = ctrlsInt;
            std::vector<int> localTgts = tgtsInt;
            auto wirePairs = createOperationWires(
                this->getNumLocalQubits(), this->getTotalNumQubits(),
                localCtrls, localTgts, totalWires);

            applyGate_MPI(wirePairs, &StateVectorCudaMPI::applyCuSVMatrixGate,
                          matrix, localCtrls, localTgts, use_adjoint);
        }
    }

    template <typename F, typename... Args>
    void applyGate_MPI(std::vector<int2> &wirePairs, F &&functor,
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
                /* int32_t* */ &nSwapBatches));

        //
        // the main loop of index bit swaps
        //
        constexpr int nLoops = 2;
        for (int loop = 0; loop < nLoops; ++loop) {
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
                        /* const int32_t */ this->getCommRank(),
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
                // execute swap
                PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerExecute(
                    /* custatevecHandle_t */ handle_.get(),
                    /* custatevecSVSwapWorkerDescriptor_t */
                    this->getSwapWorker(),
                    /* custatevecIndex_t */ 0,
                    /* custatevecIndex_t */ parameters.transferSize));
                // all internal CUDA calls are serialized on localStream
            }
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
            PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));
            if (loop == 0) {
                std::invoke(std::forward<F>(functor), this,
                            std::forward<Args>(args)...);
            }
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
            PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));
        }
        // synchronize all operations on device
        PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
        // barrier here for time measurement
        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));
    }

    void getSVExpectationValueHostMatrix(const CFP_t *matrix,
                                         const std::vector<int> &tgtsInt,
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
            /* const uint32_t */ tgtsInt.size(),
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
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgtsInt.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
    }
    /**
     * @brief Get expectation of a given host-defined matrix.
     *
     * @param matrix Host-defined row-major order gate matrix.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto getExpectationValueHostMatrix(const std::vector<CFP_t> &matrix,
                                       const std::vector<std::size_t> &tgts) {
        MPI_Datatype message_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            message_type = MPI_DOUBLE_COMPLEX;
        } else {
            message_type = MPI_COMPLEX;
        }

        std::vector<int> tgtsInt(tgts.size());
        std::vector<int> ctrlsInt;

        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        std::vector<int> totalWires(this->getTotalNumQubits(), 0);

        bool BSwapReq =
            IsIndexSwapRequired(this->getNumLocalQubits(),
                                this->getTotalNumQubits(), tgtsInt, totalWires);

        PL_MPI_IS_SUCCESS(MPI_Barrier(mpiCommunicator_));

        CFP_t local_expect, expect;
        if (!BSwapReq) {
            getSVExpectationValueHostMatrix(matrix.data(), tgtsInt,
                                            &local_expect);
        } else {
            std::vector<int> localTgts(tgtsInt);
            auto wirePairs = createOperationWires(this->getNumLocalQubits(),
                                                  this->getTotalNumQubits(),
                                                  localTgts, totalWires);
            applyGate_MPI(wirePairs,
                          &StateVectorCudaMPI::getSVExpectationValueHostMatrix,
                          matrix.data(), localTgts, &local_expect);
        }

        PL_MPI_IS_SUCCESS(MPI_Reduce(&local_expect, &expect, 1, message_type,
                                     MPI_SUM, 0, mpiCommunicator_));
        PL_MPI_IS_SUCCESS(
            MPI_Bcast(&expect, 1, message_type, 0, mpiCommunicator_));
        return expect;
    }

    /**
     * @brief Get expectation of a given host or device defined array.
     *
     * @param matrix Host or device defined row-major order gate matrix array.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto getExpectationValueDeviceMatrix(const CFP_t *matrix,
                                         const std::vector<std::size_t> &tgts) {
        MPI_Datatype message_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            message_type = MPI_DOUBLE_COMPLEX;
        } else {
            message_type = MPI_COMPLEX;
        }

        std::vector<int> tgtsInt(tgts.size());

        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        std::vector<int> totalWires(this->getTotalNumQubits(), 0);

        bool BSwapReq =
            IsIndexSwapRequired(this->getNumLocalQubits(),
                                this->getTotalNumQubits(), tgtsInt, totalWires);

        CFP_t local_expect, expect;
        if (!BSwapReq) {
            getSVExpectationValueHostMatrix(matrix, tgtsInt, local_expect);
        } else {
            std::vector<int> localTgts(tgtsInt);
            auto wirePairs = createOperationWires(this->getNumLocalQubits(),
                                                  this->getTotalNumQubits(),
                                                  localTgts, totalWires);
            applyGate_MPI(wirePairs,
                          &StateVectorCudaMPI::getSVExpectationValueHostMatrix,
                          matrix, localTgts, local_expect);
        }

        PL_MPI_IS_SUCCESS(MPI_Reduce(&local_expect, &expect, 1, message_type,
                                     MPI_SUM, 0, mpiCommunicator_));
        PL_MPI_IS_SUCCESS(
            MPI_Bcast(&expect, 1, message_type, 0, mpiCommunicator_));
        return expect;
    }
};

}; // namespace Pennylane
