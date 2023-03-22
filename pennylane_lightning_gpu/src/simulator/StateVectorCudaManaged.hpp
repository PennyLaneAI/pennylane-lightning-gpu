// Copyright 2022 Xanadu Quantum Technologies Inc. and contributors.

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
 * @file StateVectorCudaManaged.hpp
 */
#pragma once

#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuComplex.h> // cuDoubleComplex
#include <cuda.h>
#include <custatevec.h> // custatevecApplyMatrix

#include "Constant.hpp"
#include "DataBuffer.hpp"
#include "Error.hpp"
#include "StateVectorCudaBase.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::CUDA::Util;
using namespace Pennylane::CUDA;
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
class StateVectorCudaManaged
    : public StateVectorCudaBase<Precision, StateVectorCudaManaged<Precision>> {
  private:
    using BaseType = StateVectorCudaBase<Precision, StateVectorCudaManaged>;

  public:
    using CFP_t =
        typename StateVectorCudaBase<Precision,
                                     StateVectorCudaManaged<Precision>>::CFP_t;
    using GateType = CFP_t *;

    StateVectorCudaManaged() = delete;
    StateVectorCudaManaged(size_t num_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaManaged<Precision>>(
              num_qubits),
          data_buffer_{std::make_unique<DataBuffer<CFP_t>>(
              Util::exp2(num_qubits), dev_tag, true)},
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()), gate_cache_(true){};

    StateVectorCudaManaged(
        size_t num_qubits, const DevTag<int> &dev_tag, bool alloc = true,
        SharedCusvHandle cusvhandle_in = make_shared_cusv_handle(),
        SharedCublasCaller cublascaller_in = make_shared_cublas_caller(),
        SharedCusparseHandle cusparsehandle_in = make_shared_cusparse_handle())
        : StateVectorCudaBase<Precision, StateVectorCudaManaged<Precision>>(
              num_qubits, dev_tag, alloc),
          data_buffer_{std::make_unique<DataBuffer<CFP_t>>(
              Util::exp2(num_qubits), dev_tag, alloc)},
          handle_(std::move(cusvhandle_in)),
          cublascaller_(std::move(cublascaller_in)),
          cusparsehandle_(std::move(cusparsehandle_in)),
          gate_cache_(true, dev_tag) {
        BaseType::initSV();
    };

    StateVectorCudaManaged(const CFP_t *gpu_data, size_t length)
        : StateVectorCudaManaged(Util::log2(length)) {
        BaseType::CopyGpuDataToGpuIn(gpu_data, length, false);
    }

    StateVectorCudaManaged(
        const CFP_t *gpu_data, size_t length, DevTag<int> dev_tag,
        SharedCusvHandle handle_in = make_shared_cusv_handle(),
        SharedCublasCaller cublascaller_in = make_shared_cublas_caller(),
        SharedCusparseHandle cusparsehandle_in = make_shared_cusparse_handle())
        : StateVectorCudaManaged(
              Util::log2(length), dev_tag, true, std::move(handle_in),
              std::move(cublascaller_in), std::move(cusparsehandle_in)) {
        BaseType::CopyGpuDataToGpuIn(gpu_data, length, false);
    }

    StateVectorCudaManaged(const std::complex<Precision> *host_data,
                           size_t length)
        : StateVectorCudaManaged(Util::log2(length)) {
        BaseType::CopyHostDataToGpu(host_data, length, false);
    }

    StateVectorCudaManaged(const StateVectorCudaManaged &other)
        : BaseType(other.getNumQubits(), other.getDataBuffer().getDevTag()),
          handle_(other.handle_), cublascaller_(other.cublascaller_),
          cusparsehandle_(other.cusparsehandle_),
          gate_cache_(true, other.getDataBuffer().getDevTag()) {
        BaseType::CopyGpuDataToGpuIn(other);
    }

    ~StateVectorCudaManaged() = default;

    auto getData() -> CFP_t * { return data_buffer_->getData(); }
    auto getData() const -> const CFP_t * { return data_buffer_->getData(); }

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
        data_buffer_->zeroInit();

        CFP_t value_cu = cuUtil::complexToCu<std::complex<Precision>>(value);
        auto stream_id = data_buffer_->getDevTag().getStreamID();
        setBasisState_CUDA(data_buffer_->getData(), value_cu, index, async,
                           stream_id);
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
        data_buffer_->zeroInit();

        auto device_id = data_buffer_->getDevTag().getDeviceID();
        auto stream_id = data_buffer_->getDevTag().getStreamID();

        index_type num_elements = num_indices;
        DataBuffer<index_type, int> d_indices{
            static_cast<std::size_t>(num_elements), device_id, stream_id, true};
        DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(num_elements),
                                        device_id, stream_id, true};

        d_indices.CopyHostDataToGpu(indices, d_indices.getLength(), async);
        d_values.CopyHostDataToGpu(values, d_values.getLength(), async);

        setStateVector_CUDA(data_buffer_->getData(), num_elements,
                            d_values.getData(), d_indices.getData(),
                            thread_per_block, stream_id);
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
        PL_ABORT_IF_NOT(opNames.size() == wires.size(),
                        "Incompatible number of ops and wires");
        PL_ABORT_IF_NOT(opNames.size() == adjoints.size(),
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
        PL_ABORT_IF_NOT(opNames.size() == wires.size(),
                        "Incompatible number of ops and wires");
        PL_ABORT_IF_NOT(opNames.size() == adjoints.size(),
                        "Incompatible number of ops and adjoints");
        const auto num_ops = opNames.size();
        for (std::size_t op_idx = 0; op_idx < num_ops; op_idx++) {
            applyOperation(opNames[op_idx], wires[op_idx], adjoints[op_idx]);
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

  private:
    SharedCusvHandle handle_;
    SharedCublasCaller cublascaller_;
    mutable SharedCusparseHandle
        cusparsehandle_; // This member is mutable to allow lazy initialization.
    GateCache<Precision> gate_cache_;
    using ParFunc = std::function<void(const std::vector<size_t> &, bool,
                                       const std::vector<Precision> &)>;
    using FMap = std::unordered_map<std::string, ParFunc>;
    const FMap par_gates_{
        {"RX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyRX(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params[0])>(params[0]));
         }},
        {"RY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyRY(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params[0])>(params[0]));
         }},
        {"RZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyRZ(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params[0])>(params[0]));
         }},
        {"PhaseShift",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"MultiRZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyMultiRZ(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyIsingXX(
                 std::forward<decltype(wires)>(wires),
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
             BaseType::applyIsingZZ(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyCRX(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyCRY(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyCRZ(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitation",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applySingleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationPlus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applySingleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationMinus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applySingleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitation",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyDoubleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationPlus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyDoubleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationMinus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyDoubleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"ControlledPhaseShift",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyControlledPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"Rot",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyRot(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params)>(params));
         }},
        {"CRot", [&](auto &&wires, auto &&adjoint, auto &&params) {
             BaseType::applyCRot(std::forward<decltype(wires)>(wires),
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
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        int nIndexBits = BaseType::getNumQubits();

        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

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
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrlsInt.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
    }

    /**
     * @brief Apply a given host-matrix `matrix` to the statevector at qubit
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
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        int nIndexBits = BaseType::getNumQubits();

        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

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
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrlsInt.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
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

  private:
    std::unique_ptr<DataBuffer<CFP_t>> data_buffer_;
};
}; // namespace Pennylane
