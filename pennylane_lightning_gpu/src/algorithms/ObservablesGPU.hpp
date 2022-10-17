#pragma once

#include <functional>
#include <vector>

#include "StateVectorCudaManaged.hpp"

namespace Pennylane::Algorithms {

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
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
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
 * @brief Class models arbitrary Hermitian observables
 *
 */
template <typename T> class HermitianObsGPU final : public ObservableGPU<T> {
  public:
    using MatrixT = std::vector<std::complex<T>>;

  private:
    std::vector<std::complex<T>> matrix_;
    std::vector<size_t> wires_;
    inline static const MatrixHasher mh;

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
        // To avoid collisions on cached GPU data, use matrix elements to
        // uniquely identify Hermitian
        // TODO: Replace with a performant hash function
        std::ostringstream obs_stream;
        obs_stream << "Hermitian" << mh(matrix_);
        return obs_stream.str();
    }

    void applyInPlace(StateVectorCudaManaged<T> &sv) const override {
        sv.applyOperation_std(getObsName(), wires_, false, {}, matrix_);
    }
};

/**
 * @brief Class models Tensor product observables
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

/**
 * @brief Sparse representation of HamiltonianGPU<T>
 *
 * @tparam T Floating-point precision.
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

} // namespace Pennylane::Algorithms