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
 * @file Constant.hpp
 */
#pragma once

#include <array>
#include <string_view>
#include <utility>

namespace Pennylane::CUDA::cuConstant {
/**
 * @brief Constant gate names.
 */
[[maybe_unused]] constexpr std::array const_gate_names{
    std::string_view{"Identity"}, std::string_view{"PauliX"},
    std::string_view{"PauliY"},   std::string_view{"PauliZ"},
    std::string_view{"Hadamard"}, std::string_view{"T"},
    std::string_view{"S"},        std::string_view{"CNOT"},
    std::string_view{"SWAP"},     std::string_view{"CZ"},
    std::string_view{"CSWAP"},    std::string_view{"Toffoli"}};

/**
 * @brief The mapping of named gates to amount of
 * control wires they have.
 */
[[maybe_unused]] constexpr std::array cmap_gates{
    std::pair<std::string_view, size_t>{"Identity", 0},
    std::pair<std::string_view, size_t>{"PauliX", 0},
    std::pair<std::string_view, size_t>{"PauliY", 0},
    std::pair<std::string_view, size_t>{"PauliZ", 0},
    std::pair<std::string_view, size_t>{"Hadamard", 0},
    std::pair<std::string_view, size_t>{"T", 0},
    std::pair<std::string_view, size_t>{"S", 0},
    std::pair<std::string_view, size_t>{"RX", 0},
    std::pair<std::string_view, size_t>{"RY", 0},
    std::pair<std::string_view, size_t>{"RZ", 0},
    std::pair<std::string_view, size_t>{"Rot", 0},
    std::pair<std::string_view, size_t>{"PhaseShift", 0},
    std::pair<std::string_view, size_t>{"ControlledPhaseShift", 1},
    std::pair<std::string_view, size_t>{"CNOT", 1},
    std::pair<std::string_view, size_t>{"SWAP", 0},
    std::pair<std::string_view, size_t>{"CZ", 1},
    std::pair<std::string_view, size_t>{"CRX", 1},
    std::pair<std::string_view, size_t>{"CRY", 1},
    std::pair<std::string_view, size_t>{"CRZ", 1},
    std::pair<std::string_view, size_t>{"CRot", 1},
    std::pair<std::string_view, size_t>{"CSWAP", 1},
    std::pair<std::string_view, size_t>{"Toffoli", 2}};
} // namespace Pennylane::CUDA::cuConstant