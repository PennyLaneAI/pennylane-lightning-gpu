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

#include "AdjointDiff.hpp"
#include "AdjointDiffGPU.hpp"
#include "JacobianTape.hpp"

#include "DevicePool.hpp" // LightningException
#include "Error.hpp"      // LightningException
#include "StateVectorCudaManaged.hpp"
#include "StateVectorCudaRaw.hpp"
#include "StateVectorManaged.hpp"
#include "StateVectorRaw.hpp"
#include "cuGateCache.hpp"
#include "cuda_helpers.hpp"

#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

/// @cond DEV
namespace {
using namespace Pennylane::CUDA;
using namespace Pennylane::Algorithms;
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
        .def(py::init<std::size_t>())                // qubits, device
        .def(py::init<const SVType<PrecisionT> &>()) // copy ctor
        .def(py::init([](const np_arr_c &arr) {
            py::buffer_info numpyArrayInfo = arr.request();
            auto *data_ptr =
                static_cast<std::complex<PrecisionT> *>(numpyArrayInfo.ptr);
            return new SVType<PrecisionT>(data_ptr,
                                          static_cast<std::size_t>(arr.size()));
        }))
        .def(py::init([](py::handle arr,
                         std::size_t size) { // Pointer to either Managed or
                                             // Raw object type
            if constexpr (std::is_trivially_constructible_v<
                              SVType<PrecisionT>, void *, std::size_t>) {
                py::object data_ptr = arr.attr("data_ptr")();
                return new SVType<PrecisionT>(
                    reinterpret_cast<void *>(data_ptr), size);
            } else {
                return new SVType<PrecisionT>(
                    reinterpret_cast<typename SVType<PrecisionT>::CFP_t *>(
                        arr.ptr()),
                    size);
            }
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
            "Col-major order.")
        .def("GenerateSamples",
             [](StateVectorCudaManaged<PrecisionT> &sv, size_t num_wires,
                size_t num_shots) {
                 auto &&result = sv.generate_samples(num_shots);
                 const size_t ndim = 2;
                 const std::vector<size_t> shape{num_shots, num_wires};
                 constexpr auto sz = sizeof(size_t);
                 const std::vector<size_t> strides{sz * num_wires, sz};
                 // return 2-D NumPy array
                 return py::array(py::buffer_info(
                     result.data(), /* data as contiguous array  */
                     sz,            /* size of one scalar        */
                     py::format_descriptor<size_t>::format(), /* data type */
                     ndim,   /* number of dimensions      */
                     shape,  /* shape of the matrix       */
                     strides /* strides for each axis     */
                     ));
             })

        .def("DeviceToHost",
             py::overload_cast<StateVectorManaged<PrecisionT> &, bool>(
                 &SVType<PrecisionT>::CopyGpuDataToHost, py::const_),
             "Synchronize data from the GPU device to host.")
        .def("DeviceToHost",
             py::overload_cast<std::complex<PrecisionT> *, size_t, bool>(
                 &SVType<PrecisionT>::CopyGpuDataToHost, py::const_),
             "Synchronize data from the GPU device to host.")
        .def(
            "DeviceToHost",
            [](const SVType<PrecisionT> &gpu_sv, np_arr_c &cpu_sv, bool) {
                py::buffer_info numpyArrayInfo = cpu_sv.request();
                auto *data_ptr =
                    static_cast<complex<PrecisionT> *>(numpyArrayInfo.ptr);
                if (cpu_sv.size()) {
                    gpu_sv.CopyGpuDataToHost(data_ptr, cpu_sv.size());
                }
            },
            "Synchronize data from the GPU device to host.")
        .def("HostToDevice",
             py::overload_cast<const std::complex<PrecisionT> *, size_t, bool>(
                 &SVType<PrecisionT>::CopyHostDataToGpu),
             "Synchronize data from the host device to GPU.")
        .def("HostToDevice",
             py::overload_cast<const std::vector<std::complex<PrecisionT>> &,
                               bool>(&SVType<PrecisionT>::CopyHostDataToGpu),
             "Synchronize data from the host device to GPU.")
        .def(
            "HostToDevice",
            [](SVType<PrecisionT> &gpu_sv, const np_arr_c &cpu_sv, bool async) {
                const py::buffer_info numpyArrayInfo = cpu_sv.request();
                const auto *data_ptr =
                    static_cast<complex<PrecisionT> *>(numpyArrayInfo.ptr);
                const auto length =
                    static_cast<size_t>(numpyArrayInfo.shape[0]);
                if (length) {
                    gpu_sv.CopyHostDataToGpu(data_ptr, length, async);
                }
            },
            "Synchronize data from the host device to GPU.")
        .def("GetNumGPUs", &getGPUCount, "Get the number of available GPUs.")
        .def("getCurrentGPU", &getGPUIdx,
             "Get the GPU index for the statevector data.")
        .def("numQubits", &SVType<PrecisionT>::getNumQubits)
        .def("dataLength", &SVType<PrecisionT>::getLength)
        .def("resetGPU", &SVType<PrecisionT>::initSV);
}

/**
 * @brief Templated class to build object and observable bound types.
 *
 * @tparam PrecisionT Precision of the statevector data.
 * @tparam ParamT Precision of the parameter data.
 * @param m Pybind11 module.
 */
template <class PrecisionT, class ParamT>
void OpsObs_class_bindings(py::module &m) {
    using np_arr_r =
        py::array_t<ParamT, py::array::c_style | py::array::forcecast>;
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);
    //***********************************************************************//
    //                              Observable
    //***********************************************************************//

    std::string cls_name = "ObsStructGPU_C" + bitsize;
    using obs_data_var = std::variant<std::monostate, np_arr_r, np_arr_c>;
    py::class_<ObsDatum<PrecisionT>>(m, cls_name.c_str(), py::module_local())
        .def(py::init([](const std::vector<std::string> &names,
                         const std::vector<obs_data_var> &params,
                         const std::vector<std::vector<size_t>> &wires) {
            std::vector<typename ObsDatum<PrecisionT>::param_var_t> conv_params(
                params.size());
            for (size_t p_idx = 0; p_idx < params.size(); p_idx++) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        if constexpr (std::is_same_v<p_t, np_arr_c>) {
                            auto buffer = param.request();
                            auto ptr =
                                static_cast<std::complex<ParamT> *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<std::complex<ParamT>>{
                                        ptr, ptr + buffer.size};
                            }
                        } else if constexpr (std::is_same_v<p_t, np_arr_r>) {
                            auto buffer = param.request();

                            auto *ptr = static_cast<ParamT *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<ParamT>{ptr, ptr + buffer.size};
                            }
                        } else {
                            PL_ABORT(
                                "Parameter datatype not current supported");
                        }
                    },
                    params[p_idx]);
            }
            return ObsDatum<PrecisionT>(names, conv_params, wires);
        }))
        .def("__repr__",
             [](const ObsDatum<PrecisionT> &obs) {
                 using namespace Pennylane::Util;
                 std::ostringstream obs_stream;
                 std::string obs_name = obs.getObsName()[0];
                 for (size_t o = 1; o < obs.getObsName().size(); o++) {
                     if (o < obs.getObsName().size()) {
                         obs_name += " @ ";
                     }
                     obs_name += obs.getObsName()[o];
                 }
                 obs_stream << "'wires' : " << obs.getObsWires();
                 return "Observable: { 'name' : " + obs_name + ", " +
                        obs_stream.str() + " }";
             })
        .def("get_name",
             [](const ObsDatum<PrecisionT> &obs) { return obs.getObsName(); })
        .def("get_wires",
             [](const ObsDatum<PrecisionT> &obs) { return obs.getObsWires(); })
        .def("get_params", [](const ObsDatum<PrecisionT> &obs) {
            py::list params;
            for (size_t i = 0; i < obs.getObsParams().size(); i++) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        if constexpr (std::is_same_v<
                                          p_t,
                                          std::vector<std::complex<ParamT>>>) {
                            params.append(py::array_t<std::complex<ParamT>>(
                                py::cast(param)));
                        } else if constexpr (std::is_same_v<
                                                 p_t, std::vector<ParamT>>) {
                            params.append(py::array_t<ParamT>(py::cast(param)));
                        } else if constexpr (std::is_same_v<p_t,
                                                            std::monostate>) {
                            params.append(py::list{});
                        } else {
                            throw("Unsupported data type");
                        }
                    },
                    obs.getObsParams()[i]);
            }
            return params;
        });

    //***********************************************************************//
    //                              Operations
    //***********************************************************************//
    cls_name = "OpsStructGPU_C" + bitsize;
    py::class_<OpsData<PrecisionT>>(m, cls_name.c_str(), py::module_local())
        .def(py::init<
             const std::vector<std::string> &,
             const std::vector<std::vector<ParamT>> &,
             const std::vector<std::vector<size_t>> &,
             const std::vector<bool> &,
             const std::vector<std::vector<std::complex<PrecisionT>>> &>())
        .def("__repr__", [](const OpsData<PrecisionT> &ops) {
            using namespace Pennylane::Util;
            std::ostringstream ops_stream;
            for (size_t op = 0; op < ops.getSize(); op++) {
                ops_stream << "{'name': " << ops.getOpsName()[op];
                ops_stream << ", 'params': " << ops.getOpsParams()[op];
                ops_stream << ", 'inv': " << ops.getOpsInverses()[op];
                ops_stream << "}";
                if (op < ops.getSize() - 1) {
                    ops_stream << ",";
                }
            }
            return "Operations: [" + ops_stream.str() + "]";
        });

    //***********************************************************************//
    //                              Adj Jac
    //***********************************************************************//

    cls_name = "AdjointJacobianGPU_C" + bitsize;
    py::class_<AdjointJacobianGPU<PrecisionT>>(m, cls_name.c_str(),
                                               py::module_local())
        .def(py::init<>())
        .def("create_ops_list",
             [](AdjointJacobianGPU<PrecisionT> &adj,
                const std::vector<std::string> &ops_name,
                const std::vector<np_arr_r> &ops_params,
                const std::vector<std::vector<size_t>> &ops_wires,
                const std::vector<bool> &ops_inverses,
                const std::vector<np_arr_c> &ops_matrices) {
                 std::vector<std::vector<PrecisionT>> conv_params(
                     ops_params.size());
                 std::vector<std::vector<std::complex<PrecisionT>>>
                     conv_matrices(ops_matrices.size());
                 static_cast<void>(adj);
                 for (size_t op = 0; op < ops_name.size(); op++) {
                     const auto p_buffer = ops_params[op].request();
                     const auto m_buffer = ops_matrices[op].request();
                     if (p_buffer.size) {
                         const auto *const p_ptr =
                             static_cast<const ParamT *>(p_buffer.ptr);
                         conv_params[op] =
                             std::vector<ParamT>{p_ptr, p_ptr + p_buffer.size};
                     }

                     if (m_buffer.size) {
                         const auto m_ptr =
                             static_cast<const std::complex<ParamT> *>(
                                 m_buffer.ptr);
                         conv_matrices[op] = std::vector<std::complex<ParamT>>{
                             m_ptr, m_ptr + m_buffer.size};
                     }
                 }

                 return OpsData<PrecisionT>{ops_name, conv_params, ops_wires,
                                            ops_inverses, conv_matrices};
             })
        .def("adjoint_jacobian",
             &AdjointJacobianGPU<PrecisionT>::adjointJacobian)
        .def("adjoint_jacobian",
             [](AdjointJacobianGPU<PrecisionT> &adj,
                const StateVectorCudaManaged<PrecisionT> &sv,
                const std::vector<Pennylane::Algorithms::ObsDatum<PrecisionT>>
                    &observables,
                const Pennylane::Algorithms::OpsData<PrecisionT> &operations,
                const std::vector<size_t> &trainableParams, size_t num_params) {
                 std::vector<std::vector<PrecisionT>> jac(
                     observables.size(),
                     std::vector<PrecisionT>(num_params, 0));

                 adj.adjointJacobian(sv.getData(), sv.getLength(), jac,
                                     observables, operations, trainableParams);
                 return py::array_t<ParamT>(py::cast(jac));
             })
        .def("adjoint_jacobian",
             [](AdjointJacobianGPU<PrecisionT> &adj,
                const StateVectorCudaRaw<PrecisionT> &sv,
                const std::vector<Pennylane::Algorithms::ObsDatum<PrecisionT>>
                    &observables,
                const Pennylane::Algorithms::OpsData<PrecisionT> &operations,
                const std::vector<size_t> &trainableParams, size_t num_params) {
                 std::vector<std::vector<PrecisionT>> jac(
                     observables.size(),
                     std::vector<PrecisionT>(num_params, 0));

                 adj.adjointJacobian(sv.getData(), sv.getLength(), jac,
                                     observables, operations, trainableParams);
                 return py::array_t<ParamT>(py::cast(jac));
             });
}

/**
 * @brief Add C++ classes, methods and functions to Python module.
 */
PYBIND11_MODULE(lightning_gpu_qubit_ops, // NOLINT: No control over
                                         // Pybind internals
                m) {
    // Suppress doxygen autogenerated signatures

    py::options options;
    options.disable_function_signatures();
    py::register_exception<LightningException>(m, "PLException");

    m.def("device_reset", &deviceReset, "Reset all GPU devices and contexts.");
    m.def("allToAllAccess", []() {
        for (int i = 0; i < static_cast<int>(getGPUCount()); i++) {
            cudaDeviceEnablePeerAccess(i, 0);
        }
    });
    m.def("is_gpu_supported", &isCuQuantumSupported,
          py::arg("device_number") = 0,
          "Checks if the given GPU device meets the minimum architecture "
          "support for the PennyLane-Lightning-GPU device.");
    m.def("get_gpu_arch", &getGPUArch, py::arg("device_number") = 0,
          "Returns the given GPU major and minor GPU support.");

    py::class_<DevicePool<int>>(m, "DevPool")
        .def(py::init<>())
        .def("getActiveDevices", &DevicePool<int>::getActiveDevices)
        .def("isActive", &DevicePool<int>::isActive)
        .def("isInactive", &DevicePool<int>::isInactive)
        .def("getTotalDevices", &DevicePool<int>::getTotalDevices)
        .def("acquireDevice", &DevicePool<int>::acquireDevice)
        .def("releaseDevice", &DevicePool<int>::releaseDevice);

    StateVectorCuda_class_bindings<StateVectorCudaManaged, float, float>(m);
    StateVectorCuda_class_bindings<StateVectorCudaManaged, double, double>(m);
    StateVectorCuda_class_bindings<StateVectorCudaRaw, float, float>(m);
    StateVectorCuda_class_bindings<StateVectorCudaRaw, double, double>(m);
    OpsObs_class_bindings<float, float>(m);
    OpsObs_class_bindings<double, double>(m);
}

} // namespace
  /// @endcond
