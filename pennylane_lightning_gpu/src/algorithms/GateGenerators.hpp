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
 * @file GateGenerators.hpp
 */

#pragma once

#include "StateVectorCudaManaged.hpp"
#include "cuda_helpers.hpp"

#ifdef ENABLE_MPI
#include "StateVectorCudaMPI.hpp"
#endif

/// @cond DEV
namespace {
using namespace Pennylane::CUDA;
namespace cuUtil = Pennylane::CUDA::Util;

template <class CFP_t> static constexpr auto getP11_CU() -> std::vector<CFP_t> {
    return {cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}
template <class CFP_t>
static constexpr auto getP1111_CU() -> std::vector<CFP_t> {
    return {cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}
} // namespace
/// @endcond

namespace Pennylane::CUDA::Generators {

/**
 * @brief Gradient generator function associated with the RX gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorRX_GPU(SVType &sv, const std::vector<size_t> &wires,
                          const bool adj = false) {
    sv.applyPauliX(wires, adj);
}

/**
 * @brief Gradient generator function associated with the RY gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorRY_GPU(SVType &sv, const std::vector<size_t> &wires,
                          const bool adj = false) {
    sv.applyPauliY(wires, adj);
}

/**
 * @brief Gradient generator function associated with the RZ gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorRZ_GPU(SVType &sv, const std::vector<size_t> &wires,
                          const bool adj = false) {
    sv.applyPauliZ(wires, adj);
}

/**
 * @brief Gradient generator function associated with the IsingXX gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorIsingXX_GPU(SVType &sv, const std::vector<size_t> &wires,
                               const bool adj = false) {
    sv.applyGeneratorIsingXX(wires, adj);
}

/**
 * @brief Gradient generator function associated with the IsingYY gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorIsingYY_GPU(SVType &sv, const std::vector<size_t> &wires,
                               const bool adj = false) {
    sv.applyGeneratorIsingYY(wires, adj);
}

/**
 * @brief Gradient generator function associated with the IsingZZ gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorIsingZZ_GPU(SVType &sv, const std::vector<size_t> &wires,
                               const bool adj = false) {
    sv.applyGeneratorIsingZZ(wires, adj);
}

/**
 * @brief Gradient generator function associated with the PhaseShift gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorPhaseShift_GPU(SVType &sv, const std::vector<size_t> &wires,
                                  const bool adj = false) {
    sv.applyOperation("P_11", wires, adj, {0.0},
                      getP11_CU<typename SVType::CFP_t>());
}

/**
 * @brief Gradient generator function associated with the controlled RX gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorCRX_GPU(SVType &sv, const std::vector<size_t> &wires,
                           const bool adj = false) {
    sv.applyOperation("P_11", {wires.front()}, adj, {0.0},
                      getP11_CU<typename SVType::CFP_t>());
    sv.applyPauliX(std::vector<size_t>{wires.back()}, adj);
}

/**
 * @brief Gradient generator function associated with the controlled RY gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorCRY_GPU(SVType &sv, const std::vector<size_t> &wires,
                           const bool adj = false) {

    sv.applyOperation("P_11", {wires.front()}, adj, {0.0},
                      getP11_CU<typename SVType::CFP_t>());
    sv.applyPauliY(std::vector<size_t>{wires.back()}, adj);
}

/**
 * @brief Gradient generator function associated with the controlled RZ gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorCRZ_GPU(SVType &sv, const std::vector<size_t> &wires,
                           const bool adj = false) {
    sv.applyOperation("P_11", {wires.front()}, adj, {0.0},
                      getP11_CU<typename SVType::CFP_t>());
    sv.applyPauliZ(std::vector<size_t>{wires.back()}, adj);
}

/**
 * @brief Gradient generator function associated with the controlled PhaseShift
 * gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorControlledPhaseShift_GPU(SVType &sv,
                                            const std::vector<size_t> &wires,
                                            const bool adj = false) {
    sv.applyOperation("P_1111", {wires}, adj, {0.0},
                      getP1111_CU<typename SVType::CFP_t>());
}

/**
 * @brief Gradient generator function associated with the SingleExcitation gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorSingleExcitation_GPU(SVType &sv,
                                        const std::vector<size_t> &wires,
                                        const bool adj = false) {
    sv.applyGeneratorSingleExcitation(wires, adj);
}

/**
 * @brief Gradient generator function associated with the SingleExcitationMinus
 * gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorSingleExcitationMinus_GPU(SVType &sv,
                                             const std::vector<size_t> &wires,
                                             const bool adj = false) {
    sv.applyGeneratorSingleExcitationMinus(wires, adj);
}

/**
 * @brief Gradient generator function associated with the SingleExcitationPlus
 * gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorSingleExcitationPlus_GPU(SVType &sv,
                                            const std::vector<size_t> &wires,
                                            const bool adj = false) {
    sv.applyGeneratorSingleExcitationPlus(wires, adj);
}

/**
 * @brief Gradient generator function associated with the DoubleExcitation gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorDoubleExcitation_GPU(SVType &sv,
                                        const std::vector<size_t> &wires,
                                        const bool adj = false) {
    sv.applyGeneratorDoubleExcitation(wires, adj);
}

/**
 * @brief Gradient generator function associated with the DoubleExcitationMinus
 * gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorDoubleExcitationMinus_GPU(SVType &sv,
                                             const std::vector<size_t> &wires,
                                             const bool adj = false) {
    sv.applyGeneratorDoubleExcitationMinus(wires, adj);
}

/**
 * @brief Gradient generator function associated with the DoubleExcitationPlus
 * gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorDoubleExcitationPlus_GPU(SVType &sv,
                                            const std::vector<size_t> &wires,
                                            const bool adj = false) {
    sv.applyGeneratorDoubleExcitationPlus(wires, adj);
}

/**
 * @brief Gradient generator function associated with the MultiRZ gate.
 *
 * @tparam SVType StateVectorBase derived class.
 * @param sv Statevector
 * @param wires Wires to apply operation.
 * @param adj Takes adjoint of operation if true. Defaults to false.
 */
template <class SVType>
void applyGeneratorMultiRZ_GPU(SVType &sv, const std::vector<size_t> &wires,
                               const bool adj = false) {
    sv.applyGeneratorMultiRZ(wires, adj);
}

} // namespace Pennylane::CUDA::Generators