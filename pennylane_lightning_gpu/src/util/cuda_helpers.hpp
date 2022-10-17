// Adapted from JET: https://github.com/XanaduAI/jet.git
// and from Lightning: https://github.com/PennylaneAI/pennylane-lightning.git

#pragma once
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <custatevec.h>

#include "Error.hpp"
#include "Util.hpp"
#include <DevTag.hpp>

namespace Pennylane::CUDA::Util {

#ifndef CUDA_UNSAFE

/**
 * @brief Macro that throws Exception from CUDA failure error codes.
 *
 * @param err CUDA function error-code.
 */
#define PL_CUDA_IS_SUCCESS(err)                                                \
    PL_ABORT_IF_NOT(err == cudaSuccess, cudaGetErrorString(err))

#define PL_CUBLAS_IS_SUCCESS(err)                                              \
    PL_ABORT_IF_NOT(err == CUBLAS_STATUS_SUCCESS, GetCuBlasErrorString(err))

#define PL_CUSPARSE_IS_SUCCESS(err)                                            \
    PL_ABORT_IF_NOT(err == CUSPARSE_STATUS_SUCCESS, GetCuSparseErrorString(err))

/**
 * @brief Macro that throws Exception from cuQuantum failure error codes.
 *
 * @param err cuQuantum function error-code.
 */
#define PL_CUSTATEVEC_IS_SUCCESS(err)                                          \
    PL_ABORT_IF_NOT(err == CUSTATEVEC_STATUS_SUCCESS,                          \
                    GetCuStateVecErrorString(err).c_str())

#else
#define PL_CUDA_IS_SUCCESS(err)                                                \
    { static_cast<void>(err); }
#define PL_CUBLAS_IS_SUCCESS(err)                                              \
    { static_cast<void>(err); }
#define PL_CUSPARSE_IS_SUCCESS(err)                                            \
    { static_cast<void>(err); }
#define PL_CUSTATEVEC_IS_SUCCESS(err)                                          \
    { static_cast<void>(err); }
#endif

static const std::string GetCuBlasErrorString(const cublasStatus_t &err) {
    std::string result;
    switch (err) {
    case CUBLAS_STATUS_SUCCESS:
        result = "No errors";
        break;
    case CUBLAS_STATUS_NOT_INITIALIZED:
        result = "cuBLAS library was not initialized";
        break;
    case CUBLAS_STATUS_ALLOC_FAILED:
        result = "cuBLAS memory allocation failed";
        break;
    case CUBLAS_STATUS_INVALID_VALUE:
        result = "Invalid value";
        break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
        result = "CUDA device architecture mismatch";
        break;
    case CUBLAS_STATUS_MAPPING_ERROR:
        result = "cuBLAS mapping error";
        break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
        result = "Internal cuBLAS error";
        break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
        result = "Unsupported operation/device";
        break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
        result = "GPU program failed to execute";
        break;
    default:
        result = "Status not found";
    }
    return result;
}

static const std::string GetCuSparseErrorString(const cusparseStatus_t &err) {
    std::string result;
    switch (err) {
    case CUSPARSE_STATUS_SUCCESS:
        result = "No errors";
        break;
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        result = "cuSparse library was not initialized";
        break;
    case CUSPARSE_STATUS_ALLOC_FAILED:
        result = "cuSparse memory allocation failed";
        break;
    case CUSPARSE_STATUS_INVALID_VALUE:
        result = "Invalid value";
        break;
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        result = "CUDA device architecture mismatch";
        break;
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        result = "The matrix type is not supported by cuSparse";
        break;
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        result = "Internal cuBLAS error";
        break;
    case CUSPARSE_STATUS_NOT_SUPPORTED:
        result = "Unsupported operation/device";
        break;
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        result = "GPU program failed to execute";
        break;
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        result = "The resources are not sufficient to complete the operation.";
        break;
    default:
        result = "Status not found";
    }
    return result;
}

static const std::string
GetCuStateVecErrorString(const custatevecStatus_t &err) {
    std::string result;
    switch (err) {
    case CUSTATEVEC_STATUS_SUCCESS:
        result = "No errors";
        break;
    case CUSTATEVEC_STATUS_NOT_INITIALIZED:
        result = "custatevec not initialized";
        break;
    case CUSTATEVEC_STATUS_ALLOC_FAILED:
        result = "custatevec memory allocation failed";
        break;
    case CUSTATEVEC_STATUS_INVALID_VALUE:
        result = "Invalid value";
        break;
    case CUSTATEVEC_STATUS_ARCH_MISMATCH:
        result = "CUDA device architecture mismatch";
        break;
    case CUSTATEVEC_STATUS_EXECUTION_FAILED:
        result = "custatevec execution failed";
        break;
    case CUSTATEVEC_STATUS_INTERNAL_ERROR:
        result = "Internal custatevec error";
        break;
    case CUSTATEVEC_STATUS_NOT_SUPPORTED:
        result = "Unsupported operation/device";
        break;
    case CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE:
        result = "Insufficient memory for gate-application workspace";
        break;
    case CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED:
        result = "Sampler not preprocessed";
        break;
    default:
        result = "Status not found";
    }
    return result;
}

// SFINAE check for existence of real() method in complex type
template <typename CFP_t>
constexpr auto is_cxx_complex(const CFP_t &t) -> decltype(t.real(), bool()) {
    return true;
}

// Catch-all fallback for CUDA complex types
constexpr bool is_cxx_complex(...) { return false; }

inline cuFloatComplex operator-(const cuFloatComplex &a) {
    return {-a.x, -a.y};
}
inline cuDoubleComplex operator-(const cuDoubleComplex &a) {
    return {-a.x, -a.y};
}

template <class CFP_t_T, class CFP_t_U = CFP_t_T>
inline static auto Div(const CFP_t_T &a, const CFP_t_U &b) -> CFP_t_T {
    if constexpr (std::is_same_v<CFP_t_T, cuComplex> ||
                  std::is_same_v<CFP_t_T, float2>) {
        return cuCdivf(a, b);
    } else if (std::is_same_v<CFP_t_T, cuDoubleComplex> ||
               std::is_same_v<CFP_t_T, double2>) {
        return cuCdiv(a, b);
    }
}

/**
 * @brief Conjugate function for CXX & CUDA complex types
 *
 * @tparam CFP_t Complex data type. Supports std::complex<float>,
 * std::complex<double>, cuFloatComplex, cuDoubleComplex
 * @param a The given complex number
 * @return CFP_t The conjuagted complex number
 */
template <class CFP_t> inline static constexpr auto Conj(CFP_t a) -> CFP_t {
    if constexpr (std::is_same_v<CFP_t, cuComplex> ||
                  std::is_same_v<CFP_t, float2>) {
        return cuConjf(a);
    } else {
        return cuConj(a);
    }
}

/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam U Precision of real value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class Real_t, class CFP_t = cuDoubleComplex>
inline static constexpr auto ConstMultSC(Real_t a, CFP_t b) -> CFP_t {
    if constexpr (std::is_same_v<CFP_t, cuDoubleComplex>) {
        return make_cuDoubleComplex(a * b.x, a * b.y);
    } else {
        return make_cuFloatComplex(a * b.x, a * b.y);
    }
}

/**
 * @brief Utility to convert cuComplex types to std::complex types
 *
 * @tparam CFP_t cuFloatComplex or cuDoubleComplex types.
 * @param a CUDA compatible complex type.
 * @return std::complex converted a
 */
template <class CFP_t = cuDoubleComplex>
inline static constexpr auto cuToComplex(CFP_t a)
    -> std::complex<decltype(a.x)> {
    return std::complex<decltype(a.x)>{a.x, a.y};
}

/**
 * @brief Utility to convert std::complex types to cuComplex types
 *
 * @tparam CFP_t std::complex types.
 * @param a A std::complex type.
 * @return cuComplex converted a
 */
template <class CFP_t = std::complex<double>>
inline static constexpr auto complexToCu(CFP_t a) {
    if constexpr (std::is_same_v<CFP_t, std::complex<double>>) {
        return make_cuDoubleComplex(a.real(), a.imag());
    } else {
        return make_cuFloatComplex(a.real(), a.imag());
    }
}

/**
 * @brief Compile-time scalar complex times complex.
 *
 * @tparam U Precision of complex value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class CFP_t_T, class CFP_t_U = CFP_t_T>
inline static constexpr auto ConstMult(CFP_t_T a, CFP_t_U b) -> CFP_t_T {
    if constexpr (is_cxx_complex(b)) {
        return {a.real() * b.real() - a.imag() * b.imag(),
                a.real() * b.imag() + a.imag() * b.real()};
    } else {
        return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    }
}

template <class CFP_t_T, class CFP_t_U = CFP_t_T>
inline static constexpr auto ConstMultConj(CFP_t_T a, CFP_t_U b) -> CFP_t_T {
    return ConstMult(Conj(a), b);
}

/**
 * @brief Compile-time scalar complex summation.
 *
 * @tparam T Precision of complex value `a` and result.
 * @tparam U Precision of complex value `b`.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class CFP_t_T, class CFP_t_U = CFP_t_T>
inline static constexpr auto ConstSum(CFP_t_T a, CFP_t_U b) -> CFP_t_T {
    if constexpr (std::is_same_v<CFP_t_T, cuComplex> ||
                  std::is_same_v<CFP_t_T, float2>) {
        return cuCaddf(a, b);
    } else {
        return cuCadd(a, b);
    }
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class CFP_t> inline static constexpr auto ONE() -> CFP_t {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class CFP_t> inline static constexpr auto ZERO() -> CFP_t {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class CFP_t> inline static constexpr auto IMAG() -> CFP_t {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class CFP_t> inline static constexpr auto SQRT2() {
    if constexpr (std::is_same_v<CFP_t, float2> ||
                  std::is_same_v<CFP_t, cuFloatComplex>) {
        return CFP_t{0x1.6a09e6p+0F, 0}; // NOLINT: To be replaced in C++20
    } else if constexpr (std::is_same_v<CFP_t, double2> ||
                         std::is_same_v<CFP_t, cuDoubleComplex>) {
        return CFP_t{0x1.6a09e667f3bcdp+0,
                     0}; // NOLINT: To be replaced in C++20
    } else if constexpr (std::is_same_v<CFP_t, double>) {
        return 0x1.6a09e667f3bcdp+0; // NOLINT: To be replaced in C++20
    } else {
        return 0x1.6a09e6p+0F; // NOLINT: To be replaced in C++20
    }
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class CFP_t> inline static constexpr auto INVSQRT2() -> CFP_t {
    if constexpr (std::is_same_v<CFP_t, std::complex<float>> ||
                  std::is_same_v<CFP_t, std::complex<double>>) {
        return CFP_t(1 / M_SQRT2, 0);
    } else {
        return Div(CFP_t{1, 0}, SQRT2<CFP_t>());
    }
}

/**
 * @brief cuBLAS backed inner product for GPU data.
 *
 * @tparam T Complex data-type. Accepts cuFloatComplex and cuDoubleComplex
 * @param v1 Device data pointer 1
 * @param v2 Device data pointer 2
 * @param data_size Lengtyh of device data.
 * @return T Inner-product result
 */
template <class T = cuDoubleComplex, class DevTypeID = int>
inline auto innerProdC_CUDA(const T *v1, const T *v2, const int data_size,
                            int dev_id, cudaStream_t stream_id) -> T {
    T result{0.0, 0.0}; // Host result
    cublasHandle_t handle;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(dev_id));
    PL_CUBLAS_IS_SUCCESS(cublasCreate(&handle));
    PL_CUBLAS_IS_SUCCESS(cublasSetStream(handle, stream_id));

    if constexpr (std::is_same_v<T, cuFloatComplex>) {
        PL_CUBLAS_IS_SUCCESS(
            cublasCdotc(handle, data_size, v1, 1, v2, 1, &result));
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        PL_CUBLAS_IS_SUCCESS(
            cublasZdotc(handle, data_size, v1, 1, v2, 1, &result));
    }
    PL_CUBLAS_IS_SUCCESS(cublasDestroy(handle));
    return result;
}

/**
 * @brief cuBLAS backed GPU C/ZAXPY.
 *
 * @tparam CFP_t Complex data-type. Accepts std::complex<float> and
 * std::complex<double>
 * @param a scaling factor
 * @param v1 Device data pointer 1 (data to be modified)
 * @param v2 Device data pointer 2 (the result data)
 * @param data_size Length of device data.
 */
template <class CFP_t = std::complex<double>, class T = cuDoubleComplex,
          class DevTypeID = int>
inline auto scaleAndAddC_CUDA(const CFP_t a, const T *v1, T *v2,
                              const int data_size, DevTypeID dev_id,
                              cudaStream_t stream_id) {
    cublasHandle_t handle;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(dev_id));
    PL_CUBLAS_IS_SUCCESS(cublasCreate(&handle));
    PL_CUBLAS_IS_SUCCESS(cublasSetStream(handle, stream_id));

    if constexpr (std::is_same_v<T, cuComplex>) {
        const cuComplex alpha{a.real(), a.imag()};
        PL_CUBLAS_IS_SUCCESS(
            cublasCaxpy(handle, data_size, &alpha, v1, 1, v2, 1));
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        const cuDoubleComplex alpha{a.real(), a.imag()};
        PL_CUBLAS_IS_SUCCESS(
            cublasZaxpy(handle, data_size, &alpha, v1, 1, v2, 1));
    }
    PL_CUBLAS_IS_SUCCESS(cublasDestroy(handle));
}

/**
 * @brief cuBLAS backed GPU data scaling.
 *
 * @tparam CFP_t Complex data-type. Accepts std::complex<float> and
 * std::complex<double>
 * @param a scaling factor
 * @param v1 Device data pointer
 * @param data_size Length of device data.
 */
template <class CFP_t = std::complex<double>, class T = cuDoubleComplex,
          class DevTypeID = int>
inline auto scaleC_CUDA(const CFP_t a, T *v1, const int data_size,
                        DevTypeID dev_id, cudaStream_t stream_id) {

    cublasHandle_t handle;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(dev_id));
    PL_CUBLAS_IS_SUCCESS(cublasCreate(&handle));
    PL_CUBLAS_IS_SUCCESS(cublasSetStream(handle, stream_id));
    cudaDataType_t data_type;

    if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                  std::is_same_v<CFP_t, double2>) {
        data_type = CUDA_C_64F;
    } else {
        data_type = CUDA_C_32F;
    }

    PL_CUBLAS_IS_SUCCESS(cublasScalEx(handle, data_size,
                                      reinterpret_cast<const void *>(&a),
                                      data_type, v1, data_type, 1, data_type));

    PL_CUBLAS_IS_SUCCESS(cublasDestroy(handle));
}

/**
 * If T is a supported data type for gates, this expression will
 * evaluate to `true`. Otherwise, it will evaluate to `false`.
 *
 * Supported data types are `float2`, `double2`, and their aliases.
 *
 * @tparam T candidate data type
 */
template <class T>
constexpr bool is_supported_data_type =
    std::is_same_v<T, cuComplex> || std::is_same_v<T, float2> ||
    std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, double2>;

/**
 * @brief Simple overloaded method to define CUDA data type.
 *
 * @param t
 * @return cuDoubleComplex
 */
inline cuDoubleComplex getCudaType(const double &t) {
    static_cast<void>(t);
    return {};
}
/**
 * @brief Simple overloaded method to define CUDA data type.
 *
 * @param t
 * @return cuFloatComplex
 */
inline cuFloatComplex getCudaType(const float &t) {
    static_cast<void>(t);
    return {};
}

/**
 * @brief Return the number of supported CUDA capable GPU devices.
 *
 * @return std::size_t
 */
inline int getGPUCount() {
    int result;
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&result));
    return result;
}

/**
 * @brief Return the current GPU device.
 *
 * @return int
 */
inline int getGPUIdx() {
    int result;
    PL_CUDA_IS_SUCCESS(cudaGetDevice(&result));
    return result;
}

inline static void deviceReset() { PL_CUDA_IS_SUCCESS(cudaDeviceReset()); }

/**
 * @brief Checks to see if the given GPU supports the
 * PennyLane-Lightning-GPU device. Minimum supported architecture is SM 7.0.
 *
 * @param device_number GPU device index
 * @return bool
 */
static bool isCuQuantumSupported(int device_number = 0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_number);
    return deviceProp.major >= 7;
}

/**
 * @brief Get current GPU major.minor support
 *
 * @param device_number
 * @return std::pair<int,int>
 */
static std::pair<int, int> getGPUArch(int device_number = 0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_number);
    return std::make_pair(deviceProp.major, deviceProp.minor);
}

/** Chunk the data with the requested chunk size */
template <template <typename...> class Container, typename T>
auto chunkDataSize(const Container<T> &data, std::size_t chunk_size)
    -> Container<Container<T>> {
    Container<Container<T>> output;
    for (std::size_t chunk = 0; chunk < data.size(); chunk += chunk_size) {
        const auto chunk_end = std::min(data.size(), chunk + chunk_size);
        output.emplace_back(data.begin() + chunk, data.begin() + chunk_end);
    }
    return output;
}

/** Chunk the data into the requested number of chunks */
template <template <typename...> class Container, typename T>
auto chunkData(const Container<T> &data, std::size_t num_chunks)
    -> Container<Container<T>> {
    const auto rem = data.size() % num_chunks;
    const std::size_t div = static_cast<std::size_t>(data.size() / num_chunks);
    if (!div) { // Match chunks to available work
        return chunkDataSize(data, 1);
    }
    if (rem) { // We have an uneven split; ensure fair distribution
        auto output =
            chunkDataSize(Container<T>{data.begin(), data.end() - rem}, div);
        auto output_rem =
            chunkDataSize(Container<T>{data.end() - rem, data.end()}, div);
        for (std::size_t idx = 0; idx < output_rem.size(); idx++) {
            output[idx].insert(output[idx].end(), output_rem[idx].begin(),
                               output_rem[idx].end());
        }
        return output;
    }
    return chunkDataSize(data, div);
}

inline static auto pauliStringToEnum(const std::string &pauli_word)
    -> std::vector<custatevecPauli_t> {
    // Map string rep to Pauli enums
    const std::unordered_map<std::string, custatevecPauli_t> pauli_map{
        std::pair<const std::string, custatevecPauli_t>{std::string("X"),
                                                        CUSTATEVEC_PAULI_X},
        std::pair<const std::string, custatevecPauli_t>{std::string("Y"),
                                                        CUSTATEVEC_PAULI_Y},
        std::pair<const std::string, custatevecPauli_t>{std::string("Z"),
                                                        CUSTATEVEC_PAULI_Z},
        std::pair<const std::string, custatevecPauli_t>{std::string("I"),
                                                        CUSTATEVEC_PAULI_I}};

    static constexpr std::size_t num_char = 1;

    std::vector<custatevecPauli_t> output;
    output.reserve(pauli_word.size());

    for (const auto ch : pauli_word) {
        auto out = pauli_map.at(std::string(num_char, ch));
        output.push_back(out);
    }
    return output;
}

/**
 * Utility hash function for complex vectors representing matrices.
 */
struct MatrixHasher {
    template <class Precision = double>
    std::size_t
    operator()(const std::vector<std::complex<Precision>> &matrix) const {
        std::size_t hash_val = matrix.size();
        for (const auto &c_val : matrix) {
            hash_val ^= std::hash<Precision>()(c_val.real()) ^
                        std::hash<Precision>()(c_val.imag());
        }
        return hash_val;
    }
};

/** @brief `%CudaScopedDevice` uses RAII to select a CUDA device context.
 *
 * @see https://taskflow.github.io/taskflow/classtf_1_1cudaScopedDevice.html
 *
 * @note A `%CudaScopedDevice` instance cannot be moved or copied.
 *
 * @warning This class is not thread-safe.
 */
class CudaScopedDevice {
  public:
    /**
     * @brief Constructs a `%CudaScopedDevice` using a CUDA device.
     *
     *  @param device CUDA device to scope in the guard.
     */
    CudaScopedDevice(int device) {
        PL_CUDA_IS_SUCCESS(cudaGetDevice(&prev_device_));
        if (prev_device_ == device) {
            prev_device_ = -1;
        } else {
            PL_CUDA_IS_SUCCESS(cudaSetDevice(device));
        }
    }

    /**
     * @brief Destructs a `%CudaScopedDevice`, switching back to the
     * previous CUDA device context.
     */
    ~CudaScopedDevice() {
        if (prev_device_ != -1) {
            // Throwing exceptions from a destructor can be dangerous.
            // See https://isocpp.org/wiki/faq/exceptions#ctor-exceptions.
            cudaSetDevice(prev_device_);
        }
    }

    CudaScopedDevice() = delete;
    CudaScopedDevice(const CudaScopedDevice &) = delete;
    CudaScopedDevice(CudaScopedDevice &&) = delete;

  private:
    /// The previous CUDA device (or -1 if the device passed to the
    /// constructor is the current CUDA device).
    int prev_device_;
};

} // namespace Pennylane::CUDA::Util
