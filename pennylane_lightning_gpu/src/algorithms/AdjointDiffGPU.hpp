#pragma once

#include <omp.h>
#include <variant>

#include "DevicePool.hpp"
#include "JacobianTape.hpp"
#include "StateVectorCudaManaged.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::CUDA;
namespace cuUtil = Pennylane::CUDA::Util;

template <class CFP_t> static constexpr auto getP11_CU() -> std::vector<CFP_t> {
    return {cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}

template <class T = double, class SVType>
void applyGeneratorRX_GPU(SVType &sv, const std::vector<size_t> &wires,
                          const bool adj = false) {
    sv.applyPauliX(wires, adj);
}

template <class T = double, class SVType>
void applyGeneratorRY_GPU(SVType &sv, const std::vector<size_t> &wires,
                          const bool adj = false) {
    sv.applyPauliY(wires, adj);
}

template <class T = double, class SVType>
void applyGeneratorRZ_GPU(SVType &sv, const std::vector<size_t> &wires,
                          const bool adj = false) {
    sv.applyPauliZ(wires, adj);
}

template <class T = double, class SVType>
void applyGeneratorIsingXX_GPU(SVType &sv, const std::vector<size_t> &wires,
                               const bool adj = false) {
    sv.applyGeneratorIsingXX(wires, adj);
}

template <class T = double, class SVType>
void applyGeneratorIsingYY_GPU(SVType &sv, const std::vector<size_t> &wires,
                               const bool adj = false) {
    sv.applyGeneratorIsingYY(wires, adj);
}

template <class T = double, class SVType>
void applyGeneratorIsingZZ_GPU(SVType &sv, const std::vector<size_t> &wires,
                               const bool adj = false) {
    sv.applyGeneratorIsingZZ(wires, adj);
}
template <class T = double, class SVType>
void applyGeneratorPhaseShift_GPU(SVType &sv, const std::vector<size_t> &wires,
                                  const bool adj = false) {
    sv.applyOperation("P_11", wires, adj, {0.0},
                      getP11_CU<decltype(cuUtil::getCudaType(T{}))>());
}

template <class T = double, class SVType>
void applyGeneratorCRX_GPU(SVType &sv, const std::vector<size_t> &wires,
                           const bool adj = false) {
    sv.applyPauliX(std::vector<size_t>{wires.back()}, adj);
}

template <class T = double, class SVType>
void applyGeneratorCRY_GPU(SVType &sv, const std::vector<size_t> &wires,
                           const bool adj = false) {
    sv.applyPauliY(std::vector<size_t>{wires.back()}, adj);
}

template <class T = double, class SVType>
void applyGeneratorCRZ_GPU(SVType &sv, const std::vector<size_t> &wires,
                           const bool adj = false) {
    sv.applyPauliZ(std::vector<size_t>{wires.back()}, adj);
}

template <class T = double, class SVType>
void applyGeneratorControlledPhaseShift_GPU(SVType &sv,
                                            const std::vector<size_t> &wires,
                                            const bool adj = false) {
    sv.applyOperation("P_11", {wires.back()}, adj, {0.0},
                      getP11_CU<decltype(cuUtil::getCudaType(T{}))>());
}
template <class T = double, class SVType>
void applyGeneratorSingleExcitation_GPU(SVType &sv,
                                        const std::vector<size_t> &wires,
                                        const bool adj = false) {
    sv.applyGeneratorSingleExcitation(wires, adj);
}
template <class T = double, class SVType>
void applyGeneratorSingleExcitationMinus_GPU(SVType &sv,
                                             const std::vector<size_t> &wires,
                                             const bool adj = false) {
    sv.applyGeneratorSingleExcitationMinus(wires, adj);
}
template <class T = double, class SVType>
void applyGeneratorSingleExcitationPlus_GPU(SVType &sv,
                                            const std::vector<size_t> &wires,
                                            const bool adj = false) {
    sv.applyGeneratorSingleExcitationPlus(wires, adj);
}
template <class T = double, class SVType>
void applyGeneratorDoubleExcitation_GPU(SVType &sv,
                                        const std::vector<size_t> &wires,
                                        const bool adj = false) {
    sv.applyGeneratorDoubleExcitation(wires, adj);
}
template <class T = double, class SVType>
void applyGeneratorDoubleExcitationMinus_GPU(SVType &sv,
                                             const std::vector<size_t> &wires,
                                             const bool adj = false) {
    sv.applyGeneratorDoubleExcitationMinus(wires, adj);
}
template <class T = double, class SVType>
void applyGeneratorDoubleExcitationPlus_GPU(SVType &sv,
                                            const std::vector<size_t> &wires,
                                            const bool adj = false) {
    sv.applyGeneratorDoubleExcitationPlus(wires, adj);
}
template <class T = double, class SVType>
void applyGeneratorMultiRZ_GPU(SVType &sv, const std::vector<size_t> &wires,
                               const bool adj = false) {
    sv.applyGeneratorMultiRZ(wires, adj);
}
} // namespace
/// @endcond

namespace Pennylane::Algorithms {

/**
 * @brief Utility struct for observable operations used by AdjointJacobianGPU
 * class.
 *
 */
template <class T = double> class ObsDatum {
  public:
    /**
     * @brief Variant type of stored parameter data.
     */
    using param_var_t = std::variant<std::monostate, std::vector<T>,
                                     std::vector<std::complex<T>>>;

    /**
     * @brief Copy constructor for an ObsDatum object, representing a given
     * observable.
     *
     * @param obs_name Name of each operation of the observable. Tensor product
     * observables have more than one operation.
     * @param obs_params Parameters for a given observable operation ({} if
     * optional).
     * @param obs_wires Wires upon which to apply operation. Each observable
     * operation will be a separate nested list.
     */
    ObsDatum(std::vector<std::string> obs_name,
             std::vector<param_var_t> obs_params,
             std::vector<std::vector<size_t>> obs_wires)
        : obs_name_{std::move(obs_name)},
          obs_params_(std::move(obs_params)), obs_wires_{
                                                  std::move(obs_wires)} {};

    /**
     * @brief Get the number of operations in observable.
     *
     * @return size_t
     */
    [[nodiscard]] auto getSize() const -> size_t { return obs_name_.size(); }
    /**
     * @brief Get the name of the observable operations.
     *
     * @return const std::vector<std::string>&
     */
    [[nodiscard]] auto getObsName() const -> const std::vector<std::string> & {
        return obs_name_;
    }
    /**
     * @brief Get the parameters for the observable operations.
     *
     * @return const std::vector<std::vector<T>>&
     */
    [[nodiscard]] auto getObsParams() const
        -> const std::vector<param_var_t> & {
        return obs_params_;
    }
    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getObsWires() const
        -> const std::vector<std::vector<size_t>> & {
        return obs_wires_;
    }

  private:
    const std::vector<std::string> obs_name_;
    const std::vector<param_var_t> obs_params_;
    const std::vector<std::vector<size_t>> obs_wires_;
};

/**
 * @brief GPU-enabled adjoint Jacobian evaluator following the method of
 * arXiV:2009.02823
 *
 * @tparam T Floating-point precision.
 */
template <class T = double> class AdjointJacobianGPU {
  private:
    using CFP_t = decltype(cuUtil::getCudaType(T{}));
    using scalar_type_t = T;
    using GeneratorFunc = void (*)(StateVectorCudaManaged<T> &,
                                   const std::vector<size_t> &,
                                   const bool); // function pointer type

    // Holds the mapping from gate labels to associated generator functions.
    const std::unordered_map<std::string, GeneratorFunc> generator_map{
        {"RX", &::applyGeneratorRX_GPU<T, StateVectorCudaManaged<T>>},
        {"RY", &::applyGeneratorRY_GPU<T, StateVectorCudaManaged<T>>},
        {"RZ", &::applyGeneratorRZ_GPU<T, StateVectorCudaManaged<T>>},
        {"IsingXX", &::applyGeneratorIsingXX_GPU<T, StateVectorCudaManaged<T>>},
        {"IsingYY", &::applyGeneratorIsingYY_GPU<T, StateVectorCudaManaged<T>>},
        {"IsingZZ", &::applyGeneratorIsingZZ_GPU<T, StateVectorCudaManaged<T>>},
        {"CRX", &::applyGeneratorCRX_GPU<T, StateVectorCudaManaged<T>>},
        {"CRY", &::applyGeneratorCRY_GPU<T, StateVectorCudaManaged<T>>},
        {"CRZ", &::applyGeneratorCRZ_GPU<T, StateVectorCudaManaged<T>>},
        {"PhaseShift",
         ::applyGeneratorPhaseShift_GPU<T, StateVectorCudaManaged<T>>},
        {"ControlledPhaseShift",
         &applyGeneratorControlledPhaseShift_GPU<T, StateVectorCudaManaged<T>>},
        {"SingleExcitation",
         &::applyGeneratorSingleExcitation_GPU<T, StateVectorCudaManaged<T>>},
        {"SingleExcitationMinus",
         &::applyGeneratorSingleExcitationMinus_GPU<T,
                                                    StateVectorCudaManaged<T>>},
        {"SingleExcitationPlus",
         &::applyGeneratorSingleExcitationPlus_GPU<T,
                                                   StateVectorCudaManaged<T>>},
        {"DoubleExcitation",
         &::applyGeneratorDoubleExcitation_GPU<T, StateVectorCudaManaged<T>>},
        {"DoubleExcitationMinus",
         &::applyGeneratorDoubleExcitationMinus_GPU<T,
                                                    StateVectorCudaManaged<T>>},
        {"DoubleExcitationPlus",
         &::applyGeneratorDoubleExcitationPlus_GPU<T,
                                                   StateVectorCudaManaged<T>>},
        {"MultiRZ",
         &::applyGeneratorMultiRZ_GPU<T, StateVectorCudaManaged<T>>}};

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
     * @param sv1 Statevector <sv1|. Data will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param obs_index Observable index position of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     */
    inline void updateJacobian(const StateVectorCudaManaged<T> &sv1,
                               const StateVectorCudaManaged<T> &sv2,
                               std::vector<std::vector<T>> &jac,
                               T scaling_coeff, size_t obs_index,
                               size_t param_index) {
        jac[obs_index][param_index] =
            -2 * scaling_coeff *
            innerProdC_CUDA(sv1.getData(), sv2.getData(), sv1.getLength(),
                            sv1.getStream())
                .y;
    }

    /**
     * @brief Utility method to apply all operations from given
     * `%Pennylane::Algorithms::OpsData<T>` object to
     * `%StateVectorCudaManaged<T>`
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param adj Take the adjoint of the given operations.
     */
    inline void
    applyOperations(StateVectorCudaManaged<T> &state,
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
     * `%StateVectorCudaManaged<T>`.
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param op_idx Adjointed operation index to apply.
     */
    inline void
    applyOperationAdj(StateVectorCudaManaged<T> &state,
                      const Pennylane::Algorithms::OpsData<T> &operations,
                      size_t op_idx) {
        state.applyOperation(operations.getOpsName()[op_idx],
                             operations.getOpsWires()[op_idx],
                             !operations.getOpsInverses()[op_idx],
                             operations.getOpsParams()[op_idx]);
    }

    /**
     * @brief Utility method to apply a given operations from given
     * `%Pennylane::Algorithms::ObsDatum<T>` object to
     * `%StateVectorCudaManaged<T>`
     *
     * @param state Statevector to be updated.
     * @param observable Observable to apply.
     */
    inline void applyObservable(StateVectorCudaManaged<T> &state,
                                const ObsDatum<T> &observable) {
        using namespace Pennylane::Util;
        for (size_t j = 0; j < observable.getSize(); j++) {
            if (!observable.getObsParams().empty()) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        using cucomplex_t =
                            std::decay_t<decltype(state.getData())>;

                        // Apply supported gate with given params
                        if constexpr (std::is_same_v<p_t, std::vector<T>>) {
                            state.applyOperation(observable.getObsName()[j],
                                                 observable.getObsWires()[j],
                                                 false, param);
                        }
                        // Apply provided matrix
                        else if constexpr (std::is_same_v<
                                               p_t, std::vector<cucomplex_t>>) {
                            state.applyOperation(observable.getObsName()[j],
                                                 observable.getObsWires()[j],
                                                 false, {}, param);
                        } else {
                            state.applyOperation(observable.getObsName()[j],
                                                 observable.getObsWires()[j],
                                                 false);
                        }
                    },
                    observable.getObsParams()[j]);
            } else { // Offloat to SV dispatcher if no parameters provided
                state.applyOperation(observable.getObsName()[j],
                                     observable.getObsWires()[j], false);
            }
        }
    }

    /**
     * @brief OpenMP accelerated application of observables to given
     * statevectors
     *
     * @param states Vector of statevector copies, one per observable.
     * @param reference_state Reference statevector
     * @param observables Vector of observables to apply to each statevector.
     */
    inline void
    applyObservables(std::vector<StateVectorCudaManaged<T>> &states,
                     const StateVectorCudaManaged<T> &reference_state,
                     const std::vector<ObsDatum<T>> &observables) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_observables = observables.size();
        #if defined(_OPENMP)
            #pragma omp parallel default(none)                                 \
            shared(states, reference_state, observables, ex, num_observables)
        {
            #pragma omp for
        #endif
            for (size_t h_i = 0; h_i < num_observables; h_i++) {
                try {
                    states[h_i].updateData(reference_state);
                    applyObservable(states[h_i], observables[h_i]);
                } catch (...) {
                    #if defined(_OPENMP)
                        #pragma omp critical
                    #endif
                    ex = std::current_exception();
                    #if defined(_OPENMP)
                        #pragma omp cancel for
                    #endif
                }
            }
        #if defined(_OPENMP)
            if (ex) {
                #pragma omp cancel parallel
            }
        }
        #endif
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

    /**
     * @brief OpenMP accelerated application of adjoint operations to
     * statevectors.
     *
     * @param states Vector of all statevectors; 1 per observable
     * @param operations Operations list.
     * @param op_idx Index of given operation within operations list to take
     * adjoint of.
     */
    inline void
    applyOperationsAdj(std::vector<StateVectorCudaManaged<T>> &states,
                       const Pennylane::Algorithms::OpsData<T> &operations,
                       size_t op_idx) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_states = states.size();
        #if defined(_OPENMP)
            #pragma omp parallel default(none)                                 \
                shared(states, operations, op_idx, ex, num_states)
        {
            #pragma omp for
        #endif
            for (size_t obs_idx = 0; obs_idx < num_states; obs_idx++) {
                try {
                    applyOperationAdj(states[obs_idx], operations, op_idx);
                } catch (...) {
                    #if defined(_OPENMP)
                        #pragma omp critical
                    #endif
                    ex = std::current_exception();
                    #if defined(_OPENMP)
                        #pragma omp cancel for
                    #endif
                }
            }
        #if defined(_OPENMP)
            if (ex) {
                #pragma omp cancel parallel
            }
        }
        #endif
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

    /**
     * @brief Inline utility to assist with getting the Jacobian index offset.
     *
     * @param obs_index
     * @param tp_index
     * @param tp_size
     * @return size_t
     */
    inline auto getJacIndex(size_t obs_index, size_t tp_index, size_t tp_size)
        -> size_t {
        return obs_index * tp_size + tp_index;
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
    inline auto applyGenerator(StateVectorCudaManaged<T> &sv,
                               const std::string &op_name,
                               const std::vector<size_t> &wires, const bool adj)
        -> T {
        generator_map.at(op_name)(sv, wires, adj);
        return scaling_factors.at(op_name);
    }

  public:
    AdjointJacobianGPU() = default;

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
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies to a `%StateVectorCudaManaged<T>` object, with
     * one per required observable. The `operations` will be applied to the
     * internal statevector copies, with the operation indices participating in
     * the gradient calculations given in `trainableParams`, and the overall
     * number of parameters for the gradient calculation provided within
     * `num_params`. The resulting row-major ordered `jac` matrix representation
     * will be of size `trainableParams.size() * observables.size()`. OpenMP is
     * used to enable independent operations to be offloaded to threads.
     *
     * @param psi Pointer to the statevector data.
     * @param num_elements Length of the statevector data.
     * @param jac Preallocated vector for Jacobian data results.
     * @param observables Observables for which to calculate Jacobian.
     * @param operations Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void
    adjointJacobian(const CFP_t *ref_data, std::size_t length,
                    std::vector<std::vector<T>> &jac,
                    const std::vector<Pennylane::Algorithms::ObsDatum<T>> &obs,
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

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorCudaManaged<T> lambda(ref_data, length);

        // Apply given operations to statevector if requested
        if (apply_operations) {
            applyOperations(lambda, ops);
        }

        // Create observable-applied state-vectors
        std::vector<StateVectorCudaManaged<T>> H_lambda;
        for (size_t n = 0; n < num_observables; n++) {
            H_lambda.emplace_back(lambda.getNumQubits());
        }
        applyObservables(H_lambda, lambda, obs);

        StateVectorCudaManaged<T> mu(lambda.getNumQubits());

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

                    // clang-format off

                    #if defined(_OPENMP)
                        #pragma omp parallel for default(none)   \
                        shared(H_lambda, jac, mu, scalingFactor, \
                            trainableParamNumber, tp_it,         \
                            num_observables)
                    #endif

                    // clang-format on
                    for (size_t obs_idx = 0; obs_idx < num_observables;
                         obs_idx++) {
                        updateJacobian(H_lambda[obs_idx], mu, jac,
                                       scalingFactor, obs_idx,
                                       trainableParamNumber);
                    }
                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }
            applyOperationsAdj(H_lambda, ops, static_cast<size_t>(op_idx));
        }
    }

    /*template <class SVType = StateVectorCudaManaged<T>>
    void adjointJacobian(
        // const SVType &ref_state, std::vector<std::vector<T>> &jac,
        const CFP_t *ref_data, std::size_t length,
        std::vector<std::vector<T>> &jac,
        const std::vector<Pennylane::Algorithms::ObsDatum<T>> &observables,
        const Pennylane::Algorithms::OpsData<T> &operations,
        const std::vector<size_t> &trainableParams,
        bool apply_operations = false) {
        PL_ABORT_IF(trainableParams.empty(),
                    "No trainable parameters provided.");

        // Track positions within par and non-par operations
        const size_t num_observables = observables.size();
        DevicePool<int> dev_pool;

        std::vector<std::size_t> obs_indices(num_observables);
        std::iota(obs_indices.begin(), obs_indices.end(), 0);
        const auto obs_idx_per_device = chunkData(obs_indices, batch_size);

        size_t trainableParamNumber = trainableParams.size() - 1;
        size_t current_param_idx =
            operations.getNumParOps() - 1; // total number of parametric ops
        auto tp_it = trainableParams.end();

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorCudaManaged<T> lambda{ref_data, length};

        // Apply given operations to statevector if requested
        if (apply_operations) {
            applyOperations(lambda, operations);
        }

        // Create observable-applied state-vectors
        std::vector<StateVectorCudaManaged<T>> H_lambda;

        //#pragma omp parallel for
        for (auto &obs_list : obs_idx_per_device) {
            // auto idx = 0; // dev_pool.acquireDevice(); //memory allocation
            // for
            // multiple devices does not work. Temporarily assume all deata on
            // GPU 0
            for (size_t i = 0; i < obs_list.size(); i++) {
                H_lambda.emplace_back(lambda.getNumQubits()); //, idx);
            }
        }

        applyObservables(H_lambda, lambda, observables);

        SVType mu(lambda.getNumQubits());

        for (int op_idx = static_cast<int>(operations.getOpsName().size() - 1);
             op_idx >= 0; op_idx--) {

            PL_ABORT_IF(operations.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((operations.getOpsName()[op_idx] != "QubitStateVector") &&
                (operations.getOpsName()[op_idx] != "BasisState")) {
                mu.updateData(lambda);
                applyOperationAdj(lambda, operations, op_idx);

                if (operations.hasParams(op_idx)) {
                    if (std::find(trainableParams.begin(), tp_it,
                                  current_param_idx) != tp_it) {
                        const T scalingFactor =
                            applyGenerator(
                                mu, operations.getOpsName()[op_idx],
                                operations.getOpsWires()[op_idx],
                                !operations.getOpsInverses()[op_idx]) *
                            (2 * (0b1 ^ operations.getOpsInverses()[op_idx]) -
                             1);
                        // clang-format off

                        #if defined(_OPENMP)
                            #pragma omp parallel for default(none)   \
                            shared(H_lambda, jac, mu, scalingFactor, \
                                trainableParamNumber, tp_it,         \
                                num_observables)
                        #endif

                        // clang-format on
                        for (size_t obs_idx = 0; obs_idx < num_observables;
                             obs_idx++) {
                            updateJacobian(H_lambda[obs_idx], mu, jac,
                                           scalingFactor, obs_idx,
                                           trainableParamNumber);
                        }
                        trainableParamNumber--;
                        std::advance(tp_it, -1);
                    }
                    current_param_idx--;
                }

                applyOperationsAdj(H_lambda, operations,
                                   static_cast<size_t>(op_idx));
            }
        }
#pragma omp parallel for
        for (size_t i = 0; i < num_observables; i++) {
            dev_pool.releaseDevice(H_lambda[i].getDeviceID());
        }
    }*/
};

} // namespace Pennylane::Algorithms
