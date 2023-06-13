// Copyright 2022-2023 Xanadu Quantum Technologies Inc. and contributors.

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
 * @file AdjointDiffGPUMPI.hpp
 */

#pragma once

#include <memory>
#include "DevTag.hpp"
#include "GateGenerators.hpp"
#include "JacobianTape.hpp"
#include "ObservablesGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::CUDA;
namespace cuUtil = Pennylane::CUDA::Util;
namespace Generators = Pennylane::CUDA::Generators;
} // namespace
/// @endcond

namespace Pennylane::Algorithms {

/**
 * @brief GPU-enabled adjoint Jacobian evaluator following the method of
 * arXiV:2009.02823
 *
 * @tparam T Floating-point precision.
 * @tparam SVType State-vector class.
 */
template <class T = double,
          template <typename> class SVType = StateVectorCudaMPI>
class AdjointJacobianGPUMPI {
  private:
    using CFP_t = decltype(cuUtil::getCudaType(T{}));
    using scalar_type_t = T;
    using GeneratorFunc = void (*)(SVType<T> &, const std::vector<size_t> &,
                                   const bool); // function pointer type

    // Holds the mapping from gate labels to associated generator functions.
    const std::unordered_map<std::string, GeneratorFunc> generator_map{
        {"RX", &Generators::applyGeneratorRX_GPU<SVType<T>>},
        {"RY", &Generators::applyGeneratorRY_GPU<SVType<T>>},
        {"RZ", &Generators::applyGeneratorRZ_GPU<SVType<T>>},
        {"IsingXX", &Generators::applyGeneratorIsingXX_GPU<SVType<T>>},
        {"IsingYY", &Generators::applyGeneratorIsingYY_GPU<SVType<T>>},
        {"IsingZZ", &Generators::applyGeneratorIsingZZ_GPU<SVType<T>>},
        {"CRX", &Generators::applyGeneratorCRX_GPU<SVType<T>>},
        {"CRY", &Generators::applyGeneratorCRY_GPU<SVType<T>>},
        {"CRZ", &Generators::applyGeneratorCRZ_GPU<SVType<T>>},
        {"PhaseShift", &Generators::applyGeneratorPhaseShift_GPU<SVType<T>>},
        {"ControlledPhaseShift",
         &Generators::applyGeneratorControlledPhaseShift_GPU<SVType<T>>},
        {"SingleExcitation",
         &Generators::applyGeneratorSingleExcitation_GPU<SVType<T>>},
        {"SingleExcitationMinus",
         &Generators::applyGeneratorSingleExcitationMinus_GPU<SVType<T>>},
        {"SingleExcitationPlus",
         &Generators::applyGeneratorSingleExcitationPlus_GPU<SVType<T>>},
        {"DoubleExcitation",
         &Generators::applyGeneratorDoubleExcitation_GPU<SVType<T>>},
        {"DoubleExcitationMinus",
         &Generators::applyGeneratorDoubleExcitationMinus_GPU<SVType<T>>},
        {"DoubleExcitationPlus",
         &Generators::applyGeneratorDoubleExcitationPlus_GPU<SVType<T>>},
        {"MultiRZ", &Generators::applyGeneratorMultiRZ_GPU<SVType<T>>}};

    // Holds the mappings from gate labels to associated generator coefficients.
    const std::unordered_map<std::string, T> scaling_factors{
        {"RX", -static_cast<T>(0.5)},
        {"RY", -static_cast<T>(0.5)},
        {"RZ", -static_cast<T>(0.5)},
        {"IsingXX", -static_cast<T>(0.5)},
        {"IsingYY", -static_cast<T>(0.5)},
        {"IsingZZ", -static_cast<T>(0.5)},
        {"PhaseShift", static_cast<T>(1)},
        {"CRX", -static_cast<T>(0.5)},
        {"CRY", -static_cast<T>(0.5)},
        {"CRZ", -static_cast<T>(0.5)},
        {"ControlledPhaseShift", static_cast<T>(1)},
        {"SingleExcitation", -static_cast<T>(0.5)},
        {"SingleExcitationMinus", -static_cast<T>(0.5)},
        {"SingleExcitationPlus", -static_cast<T>(0.5)},
        {"DoubleExcitation", -static_cast<T>(0.5)},
        {"DoubleExcitationMinus", -static_cast<T>(0.5)},
        {"DoubleExcitationPlus", -static_cast<T>(0.5)},
        {"MultiRZ", -static_cast<T>(0.5)}};

    /**
     * @brief Utility method to update the Jacobian at a given index by
     * calculating the overlap between two given states.
     *
     * @param sv1s Statevector <sv1|. Data will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param obs_idx The index of observables of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     */
    inline void updateJacobian(const SVType<T> &sv1s, const SVType<T> &sv2,
                               std::vector<std::vector<T>> &jac,
                               T scaling_coeff, size_t obs_idx,
                               size_t param_index) {
        PL_ABORT_IF_NOT(sv1s.getDataBuffer().getDevTag().getDeviceID() ==
                            sv2.getDataBuffer().getDevTag().getDeviceID(),
                        "Data exists on different GPUs. Aborting.");
        CFP_t result;
        innerProdC_CUDA_device(sv1s.getData(), sv2.getData(), sv1s.getLength(),
                               sv1s.getDataBuffer().getDevTag().getDeviceID(),
                               sv1s.getDataBuffer().getDevTag().getStreamID(),
                               sv1s.getCublasCaller(), &result);

        auto jac_single_param =
            sv2.getMPIManager().template allreduce<CFP_t>(result, "sum");

        jac[obs_idx][param_index] = -2 * scaling_coeff * jac_single_param.y;
    }

    /**
     * @brief Utility method to apply all operations from given
     * `%Pennylane::Algorithms::OpsData<T>` object to
     * `%SVType<T>`
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param adj Take the adjoint of the given operations.
     */
    inline void
    applyOperations(SVType<T> &state,
                    const Pennylane::Algorithms::OpsData<T> &operations,
                    bool adj = false) {
        for (size_t op_idx = 0; op_idx < operations.getOpsName().size();
             op_idx++) {
            state.applyOperation(operations.getOpsName()[op_idx],
                                 operations.getOpsWires()[op_idx],
                                 operations.getOpsInverses()[op_idx] ^ adj,
                                 operations.getOpsParams()[op_idx]);
        }
    }

    /**
     * @brief Utility method to apply the adjoint indexed operation from
     * `%Pennylane::Algorithms::OpsData<T>` object to
     * `%SVType<T>`.
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param op_idx Adjointed operation index to apply.
     */
    inline void
    applyOperationAdj(SVType<T> &state,
                      const Pennylane::Algorithms::OpsData<T> &operations,
                      size_t op_idx) {
        state.applyOperation(operations.getOpsName()[op_idx],
                             operations.getOpsWires()[op_idx],
                             !operations.getOpsInverses()[op_idx],
                             operations.getOpsParams()[op_idx]);
    }

    /**
     * @brief Utility method to apply a given operations from given
     * `%ObservableGPUMPI` object to
     * `%SVType<T>`
     *
     * @param state Statevector to be updated.
     * @param observable ObservableGPUMPI to apply.
     */
    inline void applyObservable(SVType<T> &state,
                                ObservableGPUMPI<T> &observable) {
        observable.applyInPlace(state);
    }

    /**
     * @brief Applies the gate generator for a given parameteric gate. Returns
     * the associated scaling coefficient.
     *
     * @param sv Statevector data to operate upon.
     * @param op_name Name of parametric gate.
     * @param wires Wires to operate upon.
     * @param adj Indicate whether to take the adjoint of the operation.
     * @return T Generator scaling coefficient.
     */
    inline auto applyGenerator(SVType<T> &sv, const std::string &op_name,
                               const std::vector<size_t> &wires, const bool adj)
        -> T {
        generator_map.at(op_name)(sv, wires, adj);
        return scaling_factors.at(op_name);
    }

  public:
    AdjointJacobianGPUMPI() = default;

    /**
     * @brief Utility to create a given operations object.
     *
     * @param ops_name Name of operations.
     * @param ops_params Parameters for each operation in ops_name.
     * @param ops_wires Wires for each operation in ops_name.
     * @param ops_inverses Indicate whether to take adjoint of each operation in
     * ops_name.
     * @param ops_matrices Matrix definition of an operation if unsupported.
     * @return const Pennylane::Algorithms::OpsData<T>
     */
    auto createOpsData(
        const std::vector<std::string> &ops_name,
        const std::vector<std::vector<T>> &ops_params,
        const std::vector<std::vector<size_t>> &ops_wires,
        const std::vector<bool> &ops_inverses,
        const std::vector<std::vector<std::complex<T>>> &ops_matrices = {{}})
        -> Pennylane::Algorithms::OpsData<T> {
        return {ops_name, ops_params, ops_wires, ops_inverses, ops_matrices};
    }

    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates with less memory requirement.
     *
     * @param ref_sv Reference to a `%SVType<T>` object.
     * @param jac Preallocated vector for Jacobian data results.
     * @param obs ObservableGPUMPIs for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void adjointJacobian_LM(
        const SVType<T> &ref_sv, std::vector<std::vector<T>> &jac,
        const std::vector<std::shared_ptr<ObservableGPUMPI<T>>> &obs,
        const Pennylane::Algorithms::OpsData<T> &ops,
        const std::vector<size_t> &trainableParams,
        bool apply_operations = false) {
        PL_ABORT_IF(trainableParams.empty(),
                    "No trainable parameters provided.");

        const std::vector<std::string> &ops_name = ops.getOpsName();
        const size_t num_observables = obs.size();

        const size_t tp_size = trainableParams.size();
        const size_t num_param_ops = ops.getNumParOps();

        DevTag<int> dt_local(ref_sv.getDataBuffer().getDevTag());
        dt_local.refresh();
        // Create $U_{1:p}\vert \lambda \rangle$

        SVType<T> lambda_ref(dt_local, ref_sv.getNumGlobalQubits(),
                             ref_sv.getNumLocalQubits(), ref_sv.getData());
        // Apply given operations to statevector if requested
        if (apply_operations) {
            applyOperations(lambda_ref, ops);
        }

        SVType<T> mu(dt_local, lambda_ref.getNumGlobalQubits(),
                     lambda_ref.getNumLocalQubits());

        SVType<T> H_lambda(dt_local, lambda_ref.getNumGlobalQubits(),
                           lambda_ref.getNumLocalQubits(),
                           lambda_ref.getData());

        SVType<T> lambda(dt_local, lambda_ref.getNumGlobalQubits(),
                         lambda_ref.getNumLocalQubits(), lambda_ref.getData());

        for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
            lambda.updateData(lambda_ref);

            // Create observable-applied state-vectors
            H_lambda.updateData(lambda_ref);

            applyObservable(H_lambda, *obs[obs_idx]);

            size_t trainableParamNumber = tp_size - 1;
            // Track positions within par and non-par operations
            // size_t trainableParamNumber = tp_size - 1;
            size_t current_param_idx =
                num_param_ops - 1; // total number of parametric ops
            auto tp_it = trainableParams.rbegin();
            const auto tp_rend = trainableParams.rend();

            for (int op_idx = static_cast<int>(ops_name.size() - 1);
                 op_idx >= 0; op_idx--) {
                PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                            "The operation is not supported using the adjoint "
                            "differentiation method");
                if ((ops_name[op_idx] == "QubitStateVector") ||
                    (ops_name[op_idx] == "BasisState")) {
                    continue;
                }
                if (tp_it == tp_rend) {
                    break; // All done
                }

                mu.updateData(lambda);
                applyOperationAdj(lambda, ops, op_idx);

                if (ops.hasParams(op_idx)) {
                    if (current_param_idx == *tp_it) {
                        const T scalingFactor =
                            applyGenerator(mu, ops.getOpsName()[op_idx],
                                           ops.getOpsWires()[op_idx],
                                           !ops.getOpsInverses()[op_idx]) *
                            (ops.getOpsInverses()[op_idx] ? -1 : 1);
                        updateJacobian(H_lambda, mu, jac, scalingFactor,
                                       obs_idx, trainableParamNumber);
                        trainableParamNumber--;
                        ++tp_it;
                    }
                    current_param_idx--;
                }
                applyOperationAdj(H_lambda, ops, static_cast<size_t>(op_idx));
            }
        }
    }

    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates with better performance.
     *
     * @param ref_sv Reference to a `%SVType<T>` object.
     * @param jac Preallocated vector for Jacobian data results.
     * @param obs ObservableGPUMPIs for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void adjointJacobian(
        const SVType<T> &ref_sv, std::vector<std::vector<T>> &jac,
        const std::vector<std::shared_ptr<ObservableGPUMPI<T>>> &obs,
        const Pennylane::Algorithms::OpsData<T> &ops,
        const std::vector<size_t> &trainableParams,
        bool apply_operations = false) {
        PL_ABORT_IF(trainableParams.empty(),
                    "No trainable parameters provided.");

        const std::vector<std::string> &ops_name = ops.getOpsName();
        const size_t num_observables = obs.size();

        const size_t tp_size = trainableParams.size();
        const size_t num_param_ops = ops.getNumParOps();

        // Track positions within par and non-par operations
        size_t trainableParamNumber = tp_size - 1;
        size_t current_param_idx =
            num_param_ops - 1; // total number of parametric ops
        auto tp_it = trainableParams.rbegin();
        const auto tp_rend = trainableParams.rend();

        DevTag<int> dt_local(ref_sv.getDataBuffer().getDevTag());
        dt_local.refresh();
        // Create $U_{1:p}\vert \lambda \rangle$
        SVType<T> lambda(dt_local, ref_sv.getNumGlobalQubits(),
                         ref_sv.getNumLocalQubits(), ref_sv.getData());

        // Apply given operations to statevector if requested
        if (apply_operations) {
            applyOperations(lambda, ops);
        }

        lambda.getMPIManager().Barrier();

        // Create observable-applied state-vectors
        /*
        SVType<T> **H_lambda = new SVType<T> *[num_observables];

        for (size_t h_i = 0; h_i < num_observables; h_i++) {
            H_lambda[h_i] =
                new SVType<T>(dt_local, lambda.getNumGlobalQubits(),
                              lambda.getNumLocalQubits(), lambda.getData());
            applyObservable(*H_lambda[h_i], *obs[h_i]);
        }
        */
        using SVTypePtr = std::unique_ptr<SVType<T>>;
        std::unique_ptr<SVTypePtr[]> H_lambda(new SVTypePtr[num_observables]);

        for (size_t h_i = 0; h_i < num_observables; h_i++) {
            H_lambda[h_i] = std::make_unique<SVType<T>>(
            dt_local, lambda.getNumGlobalQubits(), lambda.getNumLocalQubits(),
            lambda.getData());
            applyObservable(*H_lambda[h_i], *obs[h_i]);
        }

        SVType<T> mu(dt_local, lambda.getNumGlobalQubits(),
                     lambda.getNumLocalQubits());

        for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((ops_name[op_idx] == "QubitStateVector") ||
                (ops_name[op_idx] == "BasisState")) {
                continue;
            }
            if (tp_it == tp_rend) {
                break; // All done
            }
            mu.updateData(lambda);
            applyOperationAdj(lambda, ops, op_idx);

            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {
                    const T scalingFactor =
                        applyGenerator(mu, ops.getOpsName()[op_idx],
                                       ops.getOpsWires()[op_idx],
                                       !ops.getOpsInverses()[op_idx]) *
                        (ops.getOpsInverses()[op_idx] ? -1 : 1);

                    for (size_t obs_idx = 0; obs_idx < num_observables;
                         obs_idx++) {
                        updateJacobian(*H_lambda[obs_idx], mu, jac,
                                       scalingFactor, obs_idx,
                                       trainableParamNumber);
                    }

                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }
            for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
                applyOperationAdj(*H_lambda[obs_idx], ops, op_idx);
            }
        }
        //delete[] H_lambda;
    }
};

} // namespace Pennylane::Algorithms
