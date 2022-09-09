#pragma once

#include <chrono>
#include <future>
#include <omp.h>
#include <thread>
#include <variant>

#include "DevTag.hpp"
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

/**
 * @brief A base class for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam T Floating point type
 */

template <typename T>
class ObservableGPU : public std::enable_shared_from_this<ObservableGPU<T>> {
  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<T> to compare
     */
    [[nodiscard]] virtual bool isEqual(const ObservableGPU<T> &other) const = 0;

  protected:
    ObservableGPU() = default;
    ObservableGPU(const ObservableGPU &) = default;
    ObservableGPU(ObservableGPU &&) noexcept = default;
    ObservableGPU &operator=(const ObservableGPU &) = default;
    ObservableGPU &operator=(ObservableGPU &&) noexcept = default;

  public:
    virtual ~ObservableGPU() = default;

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    virtual void applyInPlace(StateVectorCudaManaged<T> &sv) const = 0;

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
    [[nodiscard]] bool operator==(const ObservableGPU<T> &other) const {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] bool operator!=(const ObservableGPU<T> &other) const {
        return !(*this == other);
    }
};

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam T Floating point type
 */
template <typename T> class NamedObsGPU final : public ObservableGPU<T> {
  private:
    std::string obs_name_;
    std::vector<size_t> wires_;
    std::vector<T> params_;

    [[nodiscard]] bool isEqual(const ObservableGPU<T> &other) const override {
        const auto &other_cast = static_cast<const NamedObsGPU<T> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /**
     * @brief Construct a NamedObsGPU object, representing a given observable.
     *
     * @param arg1 Name of the observable.
     * @param arg2 Argument to construct wires.
     * @param arg3 Argument to construct parameters
     */
    NamedObsGPU(std::string obs_name, std::vector<size_t> wires,
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

    void applyInPlace(StateVectorCudaManaged<T> &sv) const override {
        sv.applyOperation(obs_name_, wires_, false, params_);
    }
};

/**
 * @brief Class models
 *
 */
template <typename T> class HermitianObsGPU final : public ObservableGPU<T> {
  public:
    using MatrixT = std::vector<std::complex<T>>;

  private:
    std::vector<std::complex<T>> matrix_;
    std::vector<size_t> wires_;

    [[nodiscard]] bool isEqual(const ObservableGPU<T> &other) const override {
        const auto &other_cast = static_cast<const HermitianObsGPU<T> &>(other);

        return (matrix_ == other_cast.matrix_) && (wires_ == other_cast.wires_);
    }

  public:
    /**
     * @brief Create Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    // template <typename T1>
    HermitianObsGPU(MatrixT matrix, std::vector<size_t> wires)
        : matrix_{std::move(matrix)}, wires_{std::move(wires)} {}

    [[nodiscard]] auto getMatrix() const -> const std::vector<std::complex<T>> {
        return matrix_;
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        return "Hermitian";
    }

    void applyInPlace(StateVectorCudaManaged<T> &sv) const override {
        std::string obs_name_ = "Hermitian";
        sv.applyOperation_std(obs_name_, wires_, false, {}, matrix_);
    }
};

/**
 * @brief Tensor product observable class
 */
template <typename T> class TensorProdObsGPU final : public ObservableGPU<T> {
  private:
    std::vector<std::shared_ptr<ObservableGPU<T>>> obs_;
    std::vector<size_t> all_wires_;

    [[nodiscard]] bool isEqual(const ObservableGPU<T> &other) const override {
        const auto &other_cast =
            static_cast<const TensorProdObsGPU<T> &>(other);

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
    explicit TensorProdObsGPU(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
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
        std::sort(all_wires_.begin(), all_wires_.end());
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
    create(std::initializer_list<std::shared_ptr<ObservableGPU<T>>> obs)
        -> std::shared_ptr<TensorProdObsGPU<T>> {
        return std::shared_ptr<TensorProdObsGPU<T>>{
            new TensorProdObsGPU(std::move(obs))};
    }

    static auto create(std::vector<std::shared_ptr<ObservableGPU<T>>> obs)
        -> std::shared_ptr<TensorProdObsGPU<T>> {
        return std::shared_ptr<TensorProdObsGPU<T>>{
            new TensorProdObsGPU(std::move(obs))};
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

    void applyInPlace(StateVectorCudaManaged<T> &sv) const override {
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
 * TODO: Check whether caching a sparse matrix representation can give
 * a speedup
 */
template <typename T> class HamiltonianGPU final : public ObservableGPU<T> {
  public:
    using PrecisionT = T;

  private:
    std::vector<T> coeffs_;
    std::vector<std::shared_ptr<ObservableGPU<T>>> obs_;

    [[nodiscard]] bool isEqual(const ObservableGPU<T> &other) const override {
        const auto &other_cast = static_cast<const HamiltonianGPU<T> &>(other);

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
    HamiltonianGPU(T1 &&arg1, T2 &&arg2)
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
           std::initializer_list<std::shared_ptr<ObservableGPU<T>>> arg2)
        -> std::shared_ptr<HamiltonianGPU<T>> {
        return std::shared_ptr<HamiltonianGPU<T>>(
            new HamiltonianGPU<T>{std::move(arg1), std::move(arg2)});
    }

    // to work with
    void applyInPlace(StateVectorCudaManaged<T> &sv) const override {
        using CFP_t = typename StateVectorCudaManaged<T>::CFP_t;
        DataBuffer<CFP_t, int> buffer(sv.getDataBuffer().getLength(),
                                      sv.getDataBuffer().getDevTag());
        buffer.zeroInit();

        for (size_t term_idx = 0; term_idx < coeffs_.size(); term_idx++) {
            StateVectorCudaManaged<T> tmp(sv);
            obs_[term_idx]->applyInPlace(tmp);
            scaleAndAddC_CUDA(std::complex<T>{coeffs_[term_idx], 0.0},
                              tmp.getData(), buffer.getData(), tmp.getLength(),
                              tmp.getDataBuffer().getDevTag().getDeviceID(),
                              tmp.getDataBuffer().getDevTag().getStreamID());
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
        std::sort(all_wires.begin(), all_wires.end());
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

/**
 * @brief General Hamiltonian as a sum of observables.
 *
 */

template <typename T>
class SparseHamiltonianGPU final : public ObservableGPU<T> {
  public:
    using PrecisionT = T;
    // cuSparse required index type
    using IdxT = typename std::conditional<std::is_same<T, float>::value,
                                           int32_t, int64_t>::type;

  private:
    std::vector<std::complex<T>> data_;
    std::vector<IdxT> indices_;
    std::vector<IdxT> offsets_;
    std::vector<std::size_t> wires_;

    [[nodiscard]] bool isEqual(const ObservableGPU<T> &other) const override {
        const auto &other_cast =
            static_cast<const SparseHamiltonianGPU<T> &>(other);

        if (data_ != other_cast.data_ || indices_ != other_cast.indices_ ||
            offsets_ != other_cast.offsets_) {
            return false;
        }

        return true;
    }

  public:
    /**
     * @brief Create a SparseHamiltonian from data, indices and offsets in CSR
     * format.
     *
     * @param arg1 Arguments to construct data
     * @param arg2 Arguments to construct indices
     * @param arg3 Arguments to construct offsets
     * @param arg4 Arguments to construct wires
     */
    template <typename T1, typename T2, typename T3 = T2,
              typename T4 = std::vector<std::size_t>>
    SparseHamiltonianGPU(T1 &&arg1, T2 &&arg2, T3 &&arg3, T4 &&arg4)
        : data_{std::forward<T1>(arg1)}, indices_{std::forward<T2>(arg2)},
          offsets_{std::forward<T3>(arg3)}, wires_{std::forward<T4>(arg4)} {
        PL_ASSERT(data_.size() == indices_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param arg1 Argument to construct data
     * @param arg2 Argument to construct indices
     * @param arg3 Argument to construct ofsets
     * @param arg4 Argument to construct wires
     */
    static auto create(std::initializer_list<T> arg1,
                       std::initializer_list<IdxT> arg2,
                       std::initializer_list<IdxT> arg3,
                       std::initializer_list<std::size_t> arg4)
        -> std::shared_ptr<SparseHamiltonianGPU<T>> {
        return std::shared_ptr<SparseHamiltonianGPU<T>>(
            new SparseHamiltonianGPU<T>{std::move(arg1), std::move(arg2),
                                        std::move(arg3), std::move(arg4)});
    }

    // to work with
    void applyInPlace(StateVectorCudaManaged<T> &sv) const override {
        using CFP_t = typename StateVectorCudaManaged<T>::CFP_t;

        const auto nIndexBits = sv.getNumQubits();
        const auto length = 1 << nIndexBits;
        const int64_t num_rows = static_cast<int64_t>(
            offsets_.size() -
            1); // int64_t is required for num_rows by cusparseCreateCsr
        const int64_t num_cols = static_cast<int64_t>(
            num_rows); // int64_t is required for num_cols by cusparseCreateCsr
        const int64_t nnz = static_cast<int64_t>(
            data_.size()); // int64_t is required for nnz by cusparseCreateCsr

        const CFP_t alpha = {1.0, 0.0};
        const CFP_t beta = {0.0, 0.0};

        auto device_id = sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = sv.getDataBuffer().getDevTag().getStreamID();

        DataBuffer<IdxT, int> d_csrOffsets{
            static_cast<std::size_t>(offsets_.size()), device_id, stream_id,
            true};
        DataBuffer<IdxT, int> d_columns{
            static_cast<std::size_t>(indices_.size()), device_id, stream_id,
            true};
        DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(data_.size()),
                                        device_id, stream_id, true};
        DataBuffer<CFP_t, int> d_tmp{static_cast<std::size_t>(length),
                                     device_id, stream_id, true};

        d_csrOffsets.CopyHostDataToGpu(offsets_.data(),
                                       d_csrOffsets.getLength(), false);
        d_columns.CopyHostDataToGpu(indices_.data(), d_columns.getLength(),
                                    false);
        d_values.CopyHostDataToGpu(data_.data(), d_values.getLength(), false);

        cudaDataType_t data_type;
        cusparseIndexType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSPARSE_INDEX_64I;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSPARSE_INDEX_32I;
        }

        // CUSPARSE APIs
        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t mat;
        cusparseDnVecDescr_t vecX, vecY;

        size_t bufferSize = 0;

        PL_CUSPARSE_IS_SUCCESS(cusparseCreate(&handle))

        // Create sparse matrix A in CSR format
        PL_CUSPARSE_IS_SUCCESS(cusparseCreateCsr(
            /* cusparseSpMatDescr_t* */ &mat,
            /* int64_t */ num_rows,
            /* int64_t */ num_cols,
            /* int64_t */ nnz,
            /* void* */ d_csrOffsets.getData(),
            /* void* */ d_columns.getData(),
            /* void* */ d_values.getData(),
            /* cusparseIndexType_t */ compute_type,
            /* cusparseIndexType_t */ compute_type,
            /* cusparseIndexBase_t */ CUSPARSE_INDEX_BASE_ZERO,
            /* cudaDataType */ data_type))

        // Create dense vector X
        PL_CUSPARSE_IS_SUCCESS(cusparseCreateDnVec(
            /* cusparseDnVecDescr_t* */ &vecX,
            /* int64_t */ num_cols,
            /* void* */ sv.getData(),
            /* cudaDataType */ data_type))

        // Create dense vector y
        PL_CUSPARSE_IS_SUCCESS(cusparseCreateDnVec(
            /* cusparseDnVecDescr_t* */ &vecY,
            /* int64_t */ num_rows,
            /* void* */ sv.getData(),
            /* cudaDataType */ data_type))

        // allocate an external buffer if needed
        PL_CUSPARSE_IS_SUCCESS(cusparseSpMV_bufferSize(
            /* cusparseHandle_t */ handle,
            /* cusparseOperation_t */ CUSPARSE_OPERATION_NON_TRANSPOSE,
            /* const void* */ &alpha,
            /* cusparseSpMatDescr_t */ mat,
            /* cusparseDnVecDescr_t */ vecX,
            /* const void* */ &beta,
            /* cusparseDnVecDescr_t */ vecY,
            /* cudaDataType */ data_type,
            /* cusparseSpMVAlg_t */
            CUSPARSE_SPMV_CSR_ALG1, // Can also use CUSPARSE_MV_ALG_DEFAULT
            /* size_t* */ &bufferSize))

        DataBuffer<void, int> dBuffer{bufferSize, device_id, stream_id, true};

        // execute SpMV
        PL_CUSPARSE_IS_SUCCESS(cusparseSpMV(
            /* cusparseHandle_t */ handle,
            /* cusparseOperation_t */ CUSPARSE_OPERATION_NON_TRANSPOSE,
            /* const void* */ &alpha,
            /* cusparseSpMatDescr_t */ mat,
            /* cusparseDnVecDescr_t */ vecX,
            /* const void* */ &beta,
            /* cusparseDnVecDescr_t */ vecY,
            /* cudaDataType */ data_type,
            /* cusparseSpMVAlg_t */
            CUSPARSE_SPMV_CSR_ALG1, // Can also use CUSPARSE_MV_ALG_DEFAULT
            /* void* */ dBuffer.getData()))

        // destroy matrix/vector descriptors
        PL_CUSPARSE_IS_SUCCESS(cusparseDestroySpMat(mat))
        PL_CUSPARSE_IS_SUCCESS(cusparseDestroyDnVec(vecX))
        PL_CUSPARSE_IS_SUCCESS(cusparseDestroyDnVec(vecY))
        PL_CUSPARSE_IS_SUCCESS(cusparseDestroy(handle))
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream ss;
        ss << "SparseHamiltonian: {\n'data' : ";
        for (const auto &d : data_)
            ss << d;
        ss << ",\n'indices' : ";
        for (const auto &i : indices_)
            ss << i;
        ss << ",\n'offsets' : ";
        for (const auto &o : offsets_)
            ss << o;
        ss << "\n}";
        return ss.str();
    }
    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] auto getWires() const -> std::vector<size_t> {
        return wires_;
    };
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
     * @param obs_index ObservableGPU index position of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     */
    inline void updateJacobian(const StateVectorCudaManaged<T> &sv1,
                               const StateVectorCudaManaged<T> &sv2,
                               std::vector<std::vector<T>> &jac,
                               T scaling_coeff, size_t obs_index,
                               size_t param_index) {
        PL_ABORT_IF_NOT(sv1.getDataBuffer().getDevTag().getDeviceID() ==
                            sv2.getDataBuffer().getDevTag().getDeviceID(),
                        "Data exists on different GPUs. Aborting.");

        jac[obs_index][param_index] =
            -2 * scaling_coeff *
            innerProdC_CUDA(sv1.getData(), sv2.getData(), sv1.getLength(),
                            sv1.getDataBuffer().getDevTag().getDeviceID(),
                            sv1.getDataBuffer().getDevTag().getStreamID())
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
     * `%ObservableGPU` object to
     * `%StateVectorCudaManaged<T>`
     *
     * @param state Statevector to be updated.
     * @param observable ObservableGPU to apply.
     */
    inline void applyObservable(StateVectorCudaManaged<T> &state,
                                ObservableGPU<T> &observable) {
        observable.applyInPlace(state);
    }

    /**
     * @brief OpenMP accelerated application of observables to given
     * statevectors
     *
     * @param states Vector of statevector copies, one per observable.
     * @param reference_state Reference statevector
     * @param observables Vector of observables to apply to each statevector.
     */
    inline void applyObservables(
        std::vector<StateVectorCudaManaged<T>> &states,
        const StateVectorCudaManaged<T> &reference_state,
        const std::vector<std::shared_ptr<ObservableGPU<T>>> &observables) {
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
                    applyObservable(states[h_i], *observables[h_i]);
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
     * @brief Batches the adjoint_jacobian method over the available GPUs.
     * Explicitly forbids OMP_NUM_THREADS>1 to avoid issues with std::thread
     * contention and state access issues.
     *
     * @param ref_data Pointer to the statevector data.
     * @param length Length of the statevector data.
     * @param jac Preallocated vector for Jacobian data results.
     * @param obs ObservableGPUs for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */

    void batchAdjointJacobian(
        const CFP_t *ref_data, std::size_t length,
        std::vector<std::vector<T>> &jac,
        const std::vector<std::shared_ptr<ObservableGPU<T>>> &obs,
        const Pennylane::Algorithms::OpsData<T> &ops,
        const std::vector<size_t> &trainableParams,
        bool apply_operations = false) {

        // Create a pool of available GPU devices
        DevicePool<int> dp;
        const auto num_gpus = dp.getTotalDevices();
        const auto num_chunks = num_gpus;

        // Create a vector of threads for separate GPU executions
        using namespace std::chrono_literals;
        std::vector<std::thread> threads;
        threads.reserve(num_gpus);

        // Hold results of threaded GPU executions
        std::vector<std::future<std::vector<std::vector<T>>>> futures;

        // Iterate over the chunked observables, and submit the Jacobian task
        // for execution
        for (std::size_t i = 0; i < num_chunks; i++) {
            const auto first = static_cast<std::size_t>(
                std::ceil(obs.size() * i / num_chunks));
            const auto last = static_cast<std::size_t>(
                std::ceil((obs.size() * (i + 1) / num_chunks) - 1));

            std::promise<std::vector<std::vector<T>>> jac_subset_promise;
            futures.emplace_back(jac_subset_promise.get_future());

            auto adj_lambda =
                [&](std::promise<std::vector<std::vector<T>>> j_promise,
                    std::size_t offset_first, std::size_t offset_last) {
                    // Ensure No OpenMP threads spawned;
                    // to be resolved with streams in future releases
                    omp_set_num_threads(1);
                    // Grab a GPU index, and set a device tag
                    const auto id = dp.acquireDevice();
                    DevTag<int> dt_local(id, 0);
                    dt_local.refresh();

                    // Create a sv copy on this thread and device; may not be
                    // necessary, could do in adjoint calculation directly
                    StateVectorCudaManaged<T> local_sv(ref_data, length,
                                                       dt_local);

                    // Create local store for Jacobian subset
                    std::vector<std::vector<T>> jac_local(
                        (offset_last - offset_first + 1),
                        std::vector<T>(trainableParams.size(), 0));

                    adjointJacobian(local_sv.getData(), length, jac_local,
                                    {obs.begin() + offset_first,
                                     obs.begin() + offset_last + 1},
                                    ops, trainableParams, apply_operations,
                                    dt_local);

                    j_promise.set_value(std::move(jac_local));
                    dp.releaseDevice(id);
                };
            threads.emplace_back(adj_lambda, std::move(jac_subset_promise),
                                 first, last);
        }
        /// Keep going here; ensure the new local jacs are inserted and
        /// overwrite the 0 jacs values before returning
        for (std::size_t i = 0; i < futures.size(); i++) {
            const auto first = static_cast<std::size_t>(
                std::ceil(obs.size() * i / num_chunks));

            auto jac_rows = futures[i].get();
            for (std::size_t j = 0; j < jac_rows.size(); j++) {
                jac.at(first + j) = std::move(jac_rows[j]);
            }
        }
        for (std::size_t t = 0; t < threads.size(); t++) {
            threads[t].join();
        }
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
     * @param ref_data Pointer to the statevector data.
     * @param length Length of the statevector data.
     * @param jac Preallocated vector for Jacobian data results.
     * @param obs ObservableGPUs for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void
    adjointJacobian(const CFP_t *ref_data, std::size_t length,
                    std::vector<std::vector<T>> &jac,
                    const std::vector<std::shared_ptr<ObservableGPU<T>>> &obs,
                    const Pennylane::Algorithms::OpsData<T> &ops,
                    const std::vector<size_t> &trainableParams,
                    bool apply_operations = false,
                    CUDA::DevTag<int> dev_tag = {0, 0}) {
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

        DevTag<int> dt_local(std::move(dev_tag));
        dt_local.refresh();
        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorCudaManaged<T> lambda(ref_data, length, dt_local);

        // Apply given operations to statevector if requested
        if (apply_operations) {
            applyOperations(lambda, ops);
        }

        // Create observable-applied state-vectors
        std::vector<StateVectorCudaManaged<T>> H_lambda;
        for (size_t n = 0; n < num_observables; n++) {
            H_lambda.emplace_back(lambda.getNumQubits(), dt_local);
        }
        applyObservables(H_lambda, lambda, obs);

        StateVectorCudaManaged<T> mu(lambda.getNumQubits(), dt_local);

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
};

} // namespace Pennylane::Algorithms
