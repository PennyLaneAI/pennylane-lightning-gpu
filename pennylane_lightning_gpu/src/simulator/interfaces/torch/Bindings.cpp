// Copyright 2018-2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <set>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "SVCudaTorch.hpp"
#include <torch/extension.h>

#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

/// @cond DEV
namespace {
using namespace Pennylane::CUDA;
using namespace Pennylane::CUDA::Util;
using std::complex;
using std::set;
using std::string;
using std::vector;

namespace py = pybind11;

/**
 * @brief Templated class to build all required precisions for Python module.
 *
 * @tparam SVtype Templated state-vector class type.
 * @tparam PrecisionT Precision of the statevector data.
 * @tparam ParamT Precision of the parameter data.
 * @param m Pybind11 module.
 */
template <template <typename...> class SVType, class PrecisionT, class ParamT>
void StateVectorCuda_class_bindings(py::module &m) {
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;

    /// Enable module name to be based on size of complex datatype
    /// Module naming convention:
    /// LightningGPU_<name of C++ class>_C<num bits in complex value>
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);
    std::string cls_name =
        "LightningGPU_" + SVType<PrecisionT>::getClassName() + "_C" + bitsize;

    py::class_<SVType<PrecisionT>>(m, cls_name.c_str())
        //.def(py::init<std::size_t>())                // qubits, device
        //.def(py::init<const SVType<PrecisionT> &>()) // copy ctor
        .def(py::init([](torch::Tensor &tensor) {
            return new SVType<PrecisionT>(tensor);
        }))
        .def(
            "Identity",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyIdentity(wires, adjoint);
            },
            "Apply the Identity gate.")

        .def(
            "PauliX",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyPauliX(wires, adjoint);
            },
            "Apply the PauliX gate.")

        .def(
            "PauliY",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyPauliY(wires, adjoint);
            },
            "Apply the PauliY gate.")

        .def(
            "PauliZ",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyPauliZ(wires, adjoint);
            },
            "Apply the PauliZ gate.")

        .def(
            "Hadamard",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyHadamard(wires, adjoint);
            },
            "Apply the Hadamard gate.")

        .def(
            "S",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyS(wires, adjoint);
            },
            "Apply the S gate.")

        .def(
            "T",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyT(wires, adjoint);
            },
            "Apply the T gate.")

        .def(
            "CNOT",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyCNOT(wires, adjoint);
            },
            "Apply the CNOT gate.")

        .def(
            "SWAP",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applySWAP(wires, adjoint);
            },
            "Apply the SWAP gate.")

        .def(
            "CSWAP",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyCSWAP(wires, adjoint);
            },
            "Apply the CSWAP gate.")

        .def(
            "Toffoli",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyToffoli(wires, adjoint);
            },
            "Apply the Toffoli gate.")

        .def(
            "CZ",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint,
               [[maybe_unused]] const std::vector<ParamT> &params) {
                return sv.applyCZ(wires, adjoint);
            },
            "Apply the CZ gate.")

        .def(
            "PhaseShift",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyPhaseShift(wires, adjoint, params.front());
            },
            "Apply the PhaseShift gate.")

        .def("apply",
             py::overload_cast<
                 const vector<string> &, const vector<vector<std::size_t>> &,
                 const vector<bool> &, const vector<vector<PrecisionT>> &>(
                 &SVType<PrecisionT>::applyOperation))

        .def("apply", py::overload_cast<const vector<string> &,
                                        const vector<vector<std::size_t>> &,
                                        const vector<bool> &>(
                          &SVType<PrecisionT>::applyOperation))

        .def("apply",
             py::overload_cast<const std::string &, const vector<size_t> &,
                               bool, const vector<PrecisionT> &,
                               const std::vector<std::complex<PrecisionT>> &>(
                 &SVType<PrecisionT>::applyOperation_std))

        .def(
            "ControlledPhaseShift",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyControlledPhaseShift(wires, adjoint,
                                                    params.front());
            },
            "Apply the ControlledPhaseShift gate.")

        .def(
            "RX",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyRX(wires, adjoint, params.front());
            },
            "Apply the RX gate.")

        .def(
            "RY",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyRY(wires, adjoint, params.front());
            },
            "Apply the RY gate.")

        .def(
            "RZ",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyRZ(wires, adjoint, params.front());
            },
            "Apply the RZ gate.")

        .def(
            "Rot",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyRot(wires, adjoint, params);
            },
            "Apply the Rot gate.")

        .def(
            "CRX",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyCRX(wires, adjoint, params.front());
            },
            "Apply the CRX gate.")

        .def(
            "CRY",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyCRY(wires, adjoint, params.front());
            },
            "Apply the CRY gate.")

        .def(
            "CRZ",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyCRZ(wires, adjoint, params.front());
            },
            "Apply the CRZ gate.")

        .def(
            "CRot",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires,
               bool adjoint, const std::vector<ParamT> &params) {
                return sv.applyCRot(wires, adjoint, params);
            },
            "Apply the CRot gate.")

        .def(
            "ExpectationValue",
            [](SVType<PrecisionT> &sv, const std::string &obsName,
               const std::vector<std::size_t> &wires,
               [[maybe_unused]] const std::vector<ParamT> &params,
               [[maybe_unused]] const np_arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<std::complex<ParamT>> conv_matrix;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const std::complex<ParamT> *>(m_buffer.ptr);
                    conv_matrix = std::vector<std::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                // Return the real component only
                return sv.expval(obsName, wires, params, conv_matrix).x;
            },
            "Calculate the expectation value of the given observable.")
        .def(
            "ExpectationValue",
            [](SVType<PrecisionT> &sv, const std::vector<std::string> &obsName,
               const std::vector<std::size_t> &wires,
               [[maybe_unused]] const std::vector<std::vector<ParamT>> &params,
               [[maybe_unused]] const np_arr_c &gate_matrix) {
                // internally cache by concatenation of obs names, indicated by
                // prefixed # for string
                std::string obs_concat{"#"};
                for (const auto &sub : obsName) {
                    obs_concat += sub;
                }
                const auto m_buffer = gate_matrix.request();
                std::vector<std::complex<ParamT>> conv_matrix;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const std::complex<ParamT> *>(m_buffer.ptr);
                    conv_matrix = std::vector<std::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                // Return the real component only & ignore params
                return sv
                    .expval(obs_concat, wires, std::vector<ParamT>{},
                            conv_matrix)
                    .x;
            },
            "Calculate the expectation value of the given observable.")
        .def(
            "Probability",
            [](SVType<PrecisionT> &sv, const std::vector<std::size_t> &wires) {
                return py::array_t<ParamT>(py::cast(sv.probability(wires)));
            },
            "Calculate the probabilities for given wires. Results returned in "
            "Col-major order.");
}

/**
 * @brief Add C++ classes, methods and functions to Python module.
 */
PYBIND11_MODULE(lightning_gpu_bindings_torch, // NOLINT: No control over
                                              // Pybind internals
                m) {
    // Suppress doxygen autogenerated signatures

    py::options options;
    options.disable_function_signatures();

    StateVectorCuda_class_bindings<Pennylane::SVCudaTorch, float, float>(m);
    StateVectorCuda_class_bindings<Pennylane::SVCudaTorch, double, double>(m);
}

} // namespace
  /// @endcond
