#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuComplex.h> // cuDoubleComplex
#include <cuda.h>
#include <custatevec.h> // custatevecApplyMatrix

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
          gate_cache_(true),
          gate_wires_{// Add mapping from function name to required wires.
                      {"Identity", 1},
                      {"PauliX", 1},
                      {"PauliY", 1},
                      {"PauliZ", 1},
                      {"Hadamard", 1},
                      {"T", 1},
                      {"S", 1},
                      {"RX", 1},
                      {"RY", 1},
                      {"RZ", 1},
                      {"Rot", 1},
                      {"PhaseShift", 1},
                      {"ControlledPhaseShift", 2},
                      {"CNOT", 2},
                      {"SWAP", 2},
                      {"CY", 2},
                      {"CZ", 2},
                      {"CRX", 2},
                      {"CRY", 2},
                      {"CRZ", 2},
                      {"CRot", 2},
                      {"IsingXX", 2},
                      {"IsingYY", 2},
                      {"IsingZZ", 2},
                      {"SingleExcitation", 2},
                      {"SingleExcitationMinus", 2},
                      {"SingleExcitationPlus", 2},
                      {"CSWAP", 3},
                      {"Toffoli", 3},
                      {"DoubleExcitation", 4},
                      {"DoubleExcitationMinus", 4},
                      {"DoubleExcitationPlus", 4},
                      {"OrbitalRotation", 4}},
          par_gates_{
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
                   applyPhaseShift(
                       std::forward<decltype(wires)>(wires),
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
              {"CRot",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   applyCRot(std::forward<decltype(wires)>(wires),
                             std::forward<decltype(adjoint)>(adjoint),
                             std::forward<decltype(params)>(params));
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
              {"SingleExcitation",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   applySingleExcitation(
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
              {"SingleExcitationPlus",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   applySingleExcitationPlus(
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
              {"DoubleExcitationMinus",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   applyDoubleExcitationMinus(
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
              {"OrbitalRotation",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   applyOrbitalRotation(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params[0])>(params[0]));
               }}}

    {
        BaseType::initSV();
        PL_CUSTATEVEC_IS_SUCCESS(custatevecCreate(
            /* custatevecHandle_t* */ &handle));
    };

    StateVectorCudaManaged(const CFP_t *gpu_data, size_t length)
        : StateVectorCudaManaged(Util::log2(length)) {
        BaseType::CopyGpuDataToGpuIn(gpu_data, length, false);
    }
    StateVectorCudaManaged(const std::complex<Precision> *host_data,
                           size_t length)
        : StateVectorCudaManaged(Util::log2(length)) {
        BaseType::CopyHostDataToGpu(host_data, length, false);
    }

    StateVectorCudaManaged(const StateVectorCudaManaged &other)
        : StateVectorCudaManaged(other.getNumQubits()) {
        BaseType::CopyGpuDataToGpuIn(other);
    }
    // StateVectorCudaManaged(StateVectorCudaManaged &&other) = delete;

    ~StateVectorCudaManaged() {
        PL_CUSTATEVEC_IS_SUCCESS(custatevecDestroy(
            /* custatevecHandle_t */ handle));
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
            // No op
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

    void applyIdentity(const std::vector<std::size_t> &wires, bool adjoint) {
        static_cast<void>(wires);
        static_cast<void>(adjoint);
    }

    void applyPauliX(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"PauliX"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyPauliY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"PauliY"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyPauliZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"PauliZ"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyHadamard(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"Hadamard"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyS(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"S"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyT(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"T"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyCNOT(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CNOT"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyCY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CY"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyCZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CZ"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applyToffoli(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"Toffoli"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    void applySWAP(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"SWAP"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param), {},
                              wires, adjoint);
    }
    void applyCSWAP(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"SWAP"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.front()}, {wires.begin() + 1, wires.end()},
                              adjoint);
    }
    void applyRX(const std::vector<std::size_t> &wires, bool adjoint,
                 Precision param) {
        static const std::vector<std::string> name{{"RX"}};
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
    void applyRY(const std::vector<std::size_t> &wires, bool adjoint,
                 Precision param) {
        static const std::vector<std::string> name{{"RY"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    void applyRZ(const std::vector<std::size_t> &wires, bool adjoint,
                 Precision param) {
        static const std::vector<std::string> name{{"RZ"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    void applyPhaseShift(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
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
        // applyCNOT(wires, adjoint);
        // applyCRY(wires, adjoint, param);
        // applyCNOT(wires, adjoint);
        std::array<CFP_t, 4 * 4> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t c =
                cuUtil::complexToCu<std::complex<Precision>>({std::cos(p2), 0});
            const CFP_t s =
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            const CFP_t o = cuUtil::ONE<CFP_t>();
            matrix_cu[0] = o;
            matrix_cu[5] = c;
            matrix_cu[6] = -s;
            matrix_cu[9] = s;
            matrix_cu[10] = c;
            matrix_cu[15] = o;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void
    applySingleExcitationMinus(const std::vector<std::size_t> &wires,
                               bool adjoint, Precision param) {
        // static const Precision mp2 = -param / 2;
        // applyPauliX({wires[0]}, adjoint);
        // applyPauliX({wires[1]}, adjoint); // TODO
        // applyControlledPhaseShift({wires[1], wires[0]}, adjoint, mp2);
        // applyPauliX({wires[0]}, adjoint);
        // applyPauliX({wires[1]}, adjoint);
        // applyControlledPhaseShift(wires, adjoint, mp2);
        // applyCNOT(wires, adjoint);
        // applyCRY({wires[1], wires[0]}, adjoint, param);
        // applyCNOT(wires, adjoint);
        std::array<CFP_t, 4 * 4> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t c =
                cuUtil::complexToCu<std::complex<Precision>>({std::cos(p2), 0});
            const CFP_t s =
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            const CFP_t e = cuUtil::complexToCu<std::complex<Precision>>(
                std::exp(-std::complex<Precision>(0, p2)));
            matrix_cu[0] = e;
            matrix_cu[5] = c;
            matrix_cu[6] = -s;
            matrix_cu[9] = s;
            matrix_cu[10] = c;
            matrix_cu[15] = e;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void applySingleExcitationPlus(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        // static const Precision p2 = param / 2;
        // applyPauliX({wires[0]}, adjoint);
        // applyPauliX({wires[1]}, adjoint); // TODO
        // applyControlledPhaseShift({wires[1], wires[0]}, adjoint, p2);
        // applyPauliX({wires[0]}, adjoint);
        // applyPauliX({wires[1]}, adjoint);
        // applyControlledPhaseShift(wires, adjoint, p2);
        // applyCNOT(wires, adjoint);
        // applyCRY({wires[1], wires[0]}, adjoint, param);
        // applyCNOT(wires, adjoint);
        std::array<CFP_t, 4 * 4> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t c =
                cuUtil::complexToCu<std::complex<Precision>>({std::cos(p2), 0});
            const CFP_t s =
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            const CFP_t e = cuUtil::complexToCu<std::complex<Precision>>(
                std::exp(std::complex<Precision>(0, p2)));
            matrix_cu[0] = e;
            matrix_cu[5] = c;
            matrix_cu[6] = -s;
            matrix_cu[9] = s;
            matrix_cu[10] = c;
            matrix_cu[15] = e;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void applyDoubleExcitation(const std::vector<std::size_t> &wires,
                                      bool adjoint, Precision param) {
        // This decomposition is the "upside down" version of that
        // on p17 of https://arxiv.org/abs/2104.05695
        // static const Precision p8 = param / 8;
        // applyCNOT({wires[2], wires[3]}, adjoint);
        // applyCNOT({wires[0], wires[2]});
        // applyHadamard({wires[3]}, adjoint);
        // applyHadamard({wires[0]}, adjoint);
        // applyCNOT({wires[2], wires[3]}, adjoint);
        // applyCNOT({wires[0], wires[1]}, adjoint);
        // applyRY({wires[1]}, adjoint, p8);
        // applyRY({wires[0]}, adjoint, -p8);
        // applyCNOT({wires[0], wires[3]}, adjoint);
        // applyHadamard({wires[3]}, adjoint);
        // applyCNOT({wires[3], wires[1]}, adjoint);
        // applyRY({wires[1]}, p8);
        // applyRY({wires[0]}, -p8);
        // applyCNOT({wires[2], wires[1]}, adjoint);
        // applyCNOT({wires[2], wires[0]}, adjoint);
        // applyRY({wires[1]}, adjoint, -p8);
        // applyRY({wires[0]}, adjoint, p8);
        // applyCNOT({wires[3], wires[1]}, adjoint);
        // applyHadamard({wires[3]}, adjoint);
        // applyCNOT({wires[0], wires[3]}, adjoint);
        // applyRY({wires[1]}, -p8);
        // applyRY({wires[0]}, adjoint, p8);
        // applyCNOT({wires[0], wires[1]}, adjoint);
        // applyCNOT({wires[2], wires[0]}, adjoint);
        // applyHadamard({wires[0]}, adjoint);
        // applyHadamard({wires[3]}, adjoint);
        // applyCNOT({wires[0], wires[2]}, adjoint);
        // applyCNOT({wires[2], wires[3]}, adjoint);
        std::array<CFP_t, 16 * 16> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t c =
                cuUtil::complexToCu<std::complex<Precision>>({std::cos(p2), 0});
            const CFP_t s =
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            const CFP_t o = cuUtil::ONE<CFP_t>();
            matrix_cu[0] = o;
            matrix_cu[17] = o;
            matrix_cu[34] = o;
            matrix_cu[51] = c;
            matrix_cu[60] = -s;
            matrix_cu[68] = o;
            matrix_cu[85] = o;
            matrix_cu[102] = o;
            matrix_cu[119] = o;
            matrix_cu[136] = o;
            matrix_cu[153] = o;
            matrix_cu[170] = o;
            matrix_cu[187] = o;
            matrix_cu[195] = s;
            matrix_cu[204] = c;
            matrix_cu[221] = o;
            matrix_cu[238] = o;
            matrix_cu[255] = o;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void
    applyDoubleExcitationMinus(const std::vector<std::size_t> &wires,
                               bool adjoint, Precision param) {
        std::array<CFP_t, 16 * 16> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t c =
                cuUtil::complexToCu<std::complex<Precision>>({std::cos(p2), 0});
            const CFP_t s =
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            const CFP_t e = cuUtil::complexToCu<std::complex<Precision>>(
                -std::exp(std::complex<Precision>(0, p2)));
            matrix_cu[0] = e;
            matrix_cu[17] = e;
            matrix_cu[34] = e;
            matrix_cu[51] = c;
            matrix_cu[60] = -s;
            matrix_cu[68] = e;
            matrix_cu[85] = e;
            matrix_cu[102] = e;
            matrix_cu[119] = e;
            matrix_cu[136] = e;
            matrix_cu[153] = e;
            matrix_cu[170] = e;
            matrix_cu[187] = e;
            matrix_cu[195] = s;
            matrix_cu[204] = c;
            matrix_cu[221] = e;
            matrix_cu[238] = e;
            matrix_cu[255] = e;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void applyDoubleExcitationPlus(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        std::array<CFP_t, 16 * 16> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t c =
                cuUtil::complexToCu<std::complex<Precision>>({std::cos(p2), 0});
            const CFP_t s =
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            const CFP_t e = cuUtil::complexToCu<std::complex<Precision>>(
                std::exp(std::complex<Precision>(0, p2)));
            matrix_cu[0] = e;
            matrix_cu[17] = e;
            matrix_cu[34] = e;
            matrix_cu[51] = c;
            matrix_cu[60] = -s;
            matrix_cu[68] = e;
            matrix_cu[85] = e;
            matrix_cu[102] = e;
            matrix_cu[119] = e;
            matrix_cu[136] = e;
            matrix_cu[153] = e;
            matrix_cu[170] = e;
            matrix_cu[187] = e;
            matrix_cu[195] = s;
            matrix_cu[204] = c;
            matrix_cu[221] = e;
            matrix_cu[238] = e;
            matrix_cu[255] = e;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void applyOrbitalRotation(const std::vector<std::size_t> &wires,
                                     bool adjoint, Precision param) {
        static const Precision p2 = param / 2;
        applyHadamard({wires[3]}, adjoint);
        applyHadamard({wires[2]}, adjoint);
        applyCNOT({wires[3], wires[1]}, adjoint);
        applyCNOT({wires[2], wires[0]}, adjoint);
        applyRY({wires[3]}, adjoint, p2);
        applyRY({wires[2]}, adjoint, p2);
        applyRY({wires[1]}, adjoint, p2);
        applyRY({wires[0]}, adjoint, p2);
        applyCNOT({wires[3], wires[1]}, adjoint);
        applyCNOT({wires[2], wires[0]}, adjoint);
        applyHadamard({wires[3]}, adjoint);
        applyHadamard({wires[2]}, adjoint);
    }
    inline void applyIsingXX(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        // applyCNOT(wires, adjoint);
        // applyRX({wires[0]}, adjoint, param);
        // applyCNOT(wires, adjoint);
        std::array<CFP_t, 4 * 4> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t c =
                cuUtil::complexToCu<std::complex<Precision>>({std::cos(p2), 0});
            const CFP_t neg_is =
                -IMAG<CFP_t>() *
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            matrix_cu[0] = c;
            matrix_cu[3] = neg_is;
            matrix_cu[5] = c;
            matrix_cu[6] = neg_is;
            matrix_cu[9] = neg_is;
            matrix_cu[10] = c;
            matrix_cu[12] = neg_is;
            matrix_cu[15] = c;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void applyIsingYY(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        // applyCY(wires, adjoint);
        // applyRY({wires[0]}, adjoint, param);
        // applyCY(wires, adjoint);
        std::array<CFP_t, 4 * 4> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t c =
                cuUtil::complexToCu<std::complex<Precision>>({std::cos(p2), 0});
            const CFP_t pos_is =
                IMAG<CFP_t>() *
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            const CFP_t neg_is =
                IMAG<CFP_t>() *
                cuUtil::complexToCu<std::complex<Precision>>({std::sin(p2), 0});
            matrix_cu[0] = c;
            matrix_cu[3] = is;
            matrix_cu[5] = c;
            matrix_cu[6] = neg_is;
            matrix_cu[9] = neg_is;
            matrix_cu[10] = c;
            matrix_cu[12] = is;
            matrix_cu[15] = c;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void applyIsingZZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        // applyCNOT(wires, adjoint);
        // applyRZ({wires[1]}, adjoint, param);
        // applyCNOT(wires, adjoint);
        std::array<CFP_t, 4 * 4> matrix_cu;
        { /* generate matrix */
            const Precision p2 = param / 2;
            const CFP_t pos_e = cuUtil::complexToCu<std::complex<Precision>>(
                std::exp(std::complex<Precision>(0, p2)));
            const CFP_t neg_e = cuUtil::complexToCu<std::complex<Precision>>(
                std::exp(-std::complex<Precision>(0, p2)));
            matrix_cu[0] = neg_e;
            matrix_cu[5] = pos_e;
            matrix_cu[10] = pos_e;
            matrix_cu[15] = neg_e;
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
    }
    inline void applyMultiRZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        const size_t num_wires = wires.size();
        const size_t num_rows = 1 << num_wires;
        std::vector<CFP_t> matrix_cu(num_rows * num_rows);
        { /* generate matrix */
            std::vector<Precision> eigs;
            eigs.reserve(num_rows);
            eigs.push_back(1.0);
            eigs.push_back(-1.0);
            for (size_t i = 1; i < num_wires; i++) {
                const size_t sz = eigs.size();
                for (size_t j = 0; j < sz; j++) {
                    eigs.push_back(-eigs[j]);
                }
            }

            const Precision p2 = param / 2;
            for (size_t idx = 0; idx < num_rows; idx++) {
                matrix_cu[idx * num_rows + idx] =
                    cuUtil::complexToCu<std::complex<Precision>>(
                        std::exp(-std::complex<Precision>(0, p2 * eigs[idx])));
            }
        }
        applyDeviceMatrixGate(matrix_cu.data(), {}, wires, adjoint);
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
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        PL_CUSTATEVEC_IS_SUCCESS(custatevecAbs2SumArray(
            /* custatevecHandle_t */ handle,
            /* const void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ BaseType::getNumQubits(),
            /* double* */ probabilities.data(),
            /* const int32_t* */ wires_int.data(),
            /* const uint32_t */ wires_int.size(),
            /* const int32_t* */ maskBitString,
            /* const int32_t* */ maskOrdering,
            /* const uint32_t */ maskLen));

        return probabilities;
    }

  private:
    GateCache<Precision> gate_cache_;
    const std::unordered_map<std::string, size_t> gate_wires_;
    using ParFunc = std::function<void(const std::vector<size_t> &, bool,
                                       const std::vector<Precision> &)>;
    using FMap = std::unordered_map<std::string, ParFunc>;
    const FMap par_gates_;
    custatevecHandle_t handle;

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
        int nIndexBits = BaseType::getNumQubits();

        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

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
            /* custatevecHandle_t */ handle,
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double */ local_angle,
            /* const custatevecPauli_t* */ pauli_enums.data(),
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrlsInt.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size()));
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
            /* custatevecHandle_t */ handle,
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
            /* custatevecHandle_t */ handle,
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
            /* custatevecHandle_t */ handle,
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
            /* custatevecHandle_t */ handle,
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

    /**
     * @brief Get expectation of a given host-defined matrix.
     *
     * @param matrix Host-defined row-major order gate matrix.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto getExpectationValueHostMatrix(const std::vector<CFP_t> &matrix,
                                       const std::vector<std::size_t> &tgts) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

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
            /* custatevecHandle_t */ handle,
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

        CFP_t expect;

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle,
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect,
            /* cudaDataType_t */ data_type,
            /* double* */ nullptr,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
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
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

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
            /* custatevecHandle_t */ handle,
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

        CFP_t expect;

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle,
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
        return expect;
    }
};

}; // namespace Pennylane
