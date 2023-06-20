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

#pragma once

#include <functional>
#include <vector>

#include "StateVectorCudaMPI.hpp"

namespace Pennylane::Algorithms {

/**
 * @brief A base class for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam T Floating point type
 */

template <typename T>
class ObservableGPUMPI
    : public std::enable_shared_from_this<ObservableGPUMPI<T>> {
  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<T> to compare
     */
    [[nodiscard]] virtual bool
    isEqual(const ObservableGPUMPI<T> &other) const = 0;

  protected:
    ObservableGPUMPI() = default;
    ObservableGPUMPI(const ObservableGPUMPI &) = default;
    ObservableGPUMPI(ObservableGPUMPI &&) noexcept = default;
    ObservableGPUMPI &operator=(const ObservableGPUMPI &) = default;
    ObservableGPUMPI &operator=(ObservableGPUMPI &&) noexcept = default;

  public:
    virtual ~ObservableGPUMPI() = default;

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    virtual inline void applyInPlace(StateVectorCudaMPI<T> &sv) const = 0;

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<size_t> = 0;

    /**
     * @brief Test whether this object is equal to another object
     */
    [[nodiscard]] bool operator==(const ObservableGPUMPI<T> &other) const {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] bool operator!=(const ObservableGPUMPI<T> &other) const {
        return !(*this == other);
    }
};

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam T Floating point type
 */
template <typename T> class NamedObsGPUMPI final : public ObservableGPUMPI<T> {
  private:
    std::string obs_name_;
    std::vector<size_t> wires_;
    std::vector<T> params_;

    [[nodiscard]] bool
    isEqual(const ObservableGPUMPI<T> &other) const override {
        const auto &other_cast = static_cast<const NamedObsGPUMPI<T> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /**
     * @brief Construct a NamedObsGPU object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObsGPUMPI(std::string obs_name, std::vector<size_t> wires,
                   std::vector<T> params = {})
        : obs_name_{std::move(obs_name)}, wires_{std::move(wires)},
          params_{std::move(params)} {}

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    inline void applyInPlace(StateVectorCudaMPI<T> &sv) const override {
        sv.applyOperation(obs_name_, wires_, false, params_);
    }
};

/**
 * @brief Class models arbitrary Hermitian observables
 *
 */
template <typename T>
class HermitianObsGPUMPI final : public ObservableGPUMPI<T> {
  public:
    using MatrixT = std::vector<std::complex<T>>;

  private:
    std::vector<std::complex<T>> matrix_;
    std::vector<size_t> wires_;
    inline static const MatrixHasher mh;

    [[nodiscard]] bool
    isEqual(const ObservableGPUMPI<T> &other) const override {
        const auto &other_cast =
            static_cast<const HermitianObsGPUMPI<T> &>(other);

        return (matrix_ == other_cast.matrix_) && (wires_ == other_cast.wires_);
    }

  public:
    /**
     * @brief Create Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObsGPUMPI(MatrixT matrix, std::vector<size_t> wires)
        : matrix_{std::move(matrix)}, wires_{std::move(wires)} {}

    [[nodiscard]] auto getMatrix() const -> const std::vector<std::complex<T>> {
        return matrix_;
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        // To avoid collisions on cached GPU data, use matrix elements to
        // uniquely identify Hermitian
        // TODO: Replace with a performant hash function
        std::ostringstream obs_stream;
        obs_stream << "Hermitian" << mh(matrix_);
        return obs_stream.str();
    }

    inline void applyInPlace(StateVectorCudaMPI<T> &sv) const override {
        sv.applyOperation_std(getObsName(), wires_, false, {}, matrix_);
    }
};

/**
 * @brief Class models Tensor product observables
 */
template <typename T>
class TensorProdObsGPUMPI final : public ObservableGPUMPI<T> {
  private:
    std::vector<std::shared_ptr<ObservableGPUMPI<T>>> obs_;
    std::vector<size_t> all_wires_;

    [[nodiscard]] bool
    isEqual(const ObservableGPUMPI<T> &other) const override {
        const auto &other_cast =
            static_cast<const TensorProdObsGPUMPI<T> &>(other);

        if (obs_.size() != other_cast.obs_.size()) {
            return false;
        }

        for (size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObsGPUMPI(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                if (wires.contains(wire)) {
                    PL_ABORT("All wires in observables must be disjoint.");
                }
                wires.insert(wire);
            }
        }
        all_wires_ = std::vector<size_t>(wires.begin(), wires.end());
        std::sort(all_wires_.begin(), all_wires_.end(), std::less{});
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     */
    static auto
    create(std::initializer_list<std::shared_ptr<ObservableGPUMPI<T>>> obs)
        -> std::shared_ptr<TensorProdObsGPUMPI<T>> {
        return std::shared_ptr<TensorProdObsGPUMPI<T>>{
            new TensorProdObsGPUMPI(std::move(obs))};
    }

    static auto create(std::vector<std::shared_ptr<ObservableGPUMPI<T>>> obs)
        -> std::shared_ptr<TensorProdObsGPUMPI<T>> {
        return std::shared_ptr<TensorProdObsGPUMPI<T>>{
            new TensorProdObsGPUMPI(std::move(obs))};
    }

    /**
     * @brief Get the number of operations in observable.
     *
     * @return size_t
     */
    [[nodiscard]] auto getSize() const -> size_t { return obs_.size(); }

    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return all_wires_;
    }

    inline void applyInPlace(StateVectorCudaMPI<T> &sv) const override {
        for (const auto &ob : obs_) {
            ob->applyInPlace(sv);
        }
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream obs_stream;
        const auto obs_size = obs_.size();
        for (size_t idx = 0; idx < obs_size; idx++) {
            obs_stream << obs_[idx]->getObsName();
            if (idx != obs_size - 1) {
                obs_stream << " @ ";
            }
        }
        return obs_stream.str();
    }
};

/**
 * @brief General Hamiltonian as a sum of observables.
 *
 */
template <typename T>
class HamiltonianGPUMPI final : public ObservableGPUMPI<T> {
  public:
    using PrecisionT = T;

  private:
    std::vector<T> coeffs_;
    std::vector<std::shared_ptr<ObservableGPUMPI<T>>> obs_;

    [[nodiscard]] bool
    isEqual(const ObservableGPUMPI<T> &other) const override {
        const auto &other_cast =
            static_cast<const HamiltonianGPUMPI<T> &>(other);

        if (coeffs_ != other_cast.coeffs_) {
            return false;
        }

        for (size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param arg1 Arguments to construct coefficients
     * @param arg2 Arguments to construct observables
     */
    template <typename T1, typename T2>
    HamiltonianGPUMPI(T1 &&arg1, T2 &&arg2)
        : coeffs_{std::forward<T1>(arg1)}, obs_{std::forward<T2>(arg2)} {
        PL_ASSERT(coeffs_.size() == obs_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param arg1 Argument to construct coefficients
     * @param arg2 Argument to construct terms
     */
    static auto
    create(std::initializer_list<T> arg1,
           std::initializer_list<std::shared_ptr<ObservableGPUMPI<T>>> arg2)
        -> std::shared_ptr<HamiltonianGPUMPI<T>> {
        return std::shared_ptr<HamiltonianGPUMPI<T>>(
            new HamiltonianGPUMPI<T>{std::move(arg1), std::move(arg2)});
    }

    // to work with
    inline void applyInPlace(StateVectorCudaMPI<T> &sv) const override {
        using CFP_t = typename StateVectorCudaMPI<T>::CFP_t;
        DataBuffer<CFP_t, int> buffer(sv.getDataBuffer().getLength(),
                                      sv.getDataBuffer().getDevTag());
        buffer.zeroInit();

        for (size_t term_idx = 0; term_idx < coeffs_.size(); term_idx++) {
            DevTag<int> dt_local(sv.getDataBuffer().getDevTag());
            dt_local.refresh();
            StateVectorCudaMPI<T> tmp(dt_local, sv.getNumGlobalQubits(),
                                      sv.getNumLocalQubits(), sv.getData());
            obs_[term_idx]->applyInPlace(tmp);
            scaleAndAddC_CUDA(std::complex<T>{coeffs_[term_idx], 0.0},
                              tmp.getData(), buffer.getData(), tmp.getLength(),
                              tmp.getDataBuffer().getDevTag().getDeviceID(),
                              tmp.getDataBuffer().getDevTag().getStreamID(),
                              tmp.getCublasCaller());
        }

        sv.CopyGpuDataToGpuIn(buffer.getData(), buffer.getLength());
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            wires.insert(ob_wires.begin(), ob_wires.end());
        }
        auto all_wires = std::vector<size_t>(wires.begin(), wires.end());
        std::sort(all_wires.begin(), all_wires.end(), std::less{});
        return all_wires;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {

        using Pennylane::Util::operator<<;
        std::ostringstream ss;
        ss << "Hamiltonian: { 'coeffs' : " << coeffs_ << ", 'observables' : [";
        const auto term_size = coeffs_.size();
        for (size_t t = 0; t < term_size; t++) {
            ss << obs_[t]->getObsName();
            if (t != term_size - 1) {
                ss << ", ";
            }
        }
        ss << "]}";
        return ss.str();
    }
};

} // namespace Pennylane::Algorithms
