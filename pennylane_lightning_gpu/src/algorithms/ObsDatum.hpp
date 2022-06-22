#pragma once

#include <variant>

#include "StateVectorCudaManaged.hpp"

namespace Pennylane::Algorithms {
/**
 * @brief Utility class for observable operations used by AdjointJacobianGPU
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

} // namespace Pennylane::Algorithms