#include <complex>
#include <vector>

#include "AdjointDiffGPU.hpp"
#include "StateVectorCudaManaged.hpp"
#include "StateVectorCudaRaw.hpp"
#include "StateVectorManaged.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

namespace {
using namespace Pennylane;
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool isApproxEqual(
    const std::vector<Data_t> &data1, const std::vector<Data_t> &data2,
    const typename Data_t::value_type eps =
        std::numeric_limits<typename Data_t::value_type>::epsilon() * 100) {
    if (data1.size() != data2.size()) {
        return false;
    }

    for (size_t i = 0; i < data1.size(); i++) {
        if (data1[i].real() != Approx(data2[i].real()).epsilon(eps) ||
            data1[i].imag() != Approx(data2[i].imag()).epsilon(eps)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool
isApproxEqual(const Data_t &data1, const Data_t &data2,
              typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    return !(data1.real() != Approx(data2.real()).epsilon(eps) ||
             data1.imag() != Approx(data2.imag()).epsilon(eps));
}

/**
 * @brief Multiplies every value in a dataset by a given complex scalar value.
 *
 * @tparam Data_t Precision of complex data type. Supports float and double
 * data.
 * @param data Data to be scaled.
 * @param scalar Scalar value.
 */
template <class Data_t>
void scaleVector(std::vector<std::complex<Data_t>> &data,
                 std::complex<Data_t> scalar) {
    std::transform(
        data.begin(), data.end(), data.begin(),
        [scalar](const std::complex<Data_t> &c) { return c * scalar; });
}

/**
 * @brief Utility data-structure to assist with testing StateVectorCudaManaged
 * class
 *
 * @tparam fp_t Floating-point type. Supported options: float, double
 */
template <typename fp_t> struct SVDataGPU {
    std::size_t num_qubits_;
    StateVectorManaged<fp_t> sv;
    StateVectorCudaManaged<fp_t> cuda_sv;

    SVDataGPU() = delete;

    SVDataGPU(std::size_t num_qubits)
        : num_qubits_{num_qubits}, sv{num_qubits}, cuda_sv{num_qubits} {
        cuda_sv.initSV();
    }
    SVDataGPU(std::size_t num_qubits,
              const std::vector<std::complex<fp_t>> &cdata_input)
        : num_qubits_{num_qubits}, sv{cdata_input}, cuda_sv{
                                                        cdata_input.data(),
                                                        cdata_input.size()} {}

    ~SVDataGPU() {}
};

/**
 * @brief Utility data-structure to assist with testing StateVectorCudaManaged
 * class
 *
 * @tparam fp_t Floating-point type. Supported options: float, double
 */
template <typename fp_t> struct SVDataGPURaw {
    std::size_t num_qubits_;
    StateVectorManaged<fp_t> sv;
    StateVectorCudaManaged<fp_t> cuda_sv;
    StateVectorCudaRaw<fp_t> cuda_sv_raw;

    SVDataGPURaw() = delete;

    SVDataGPURaw(std::size_t num_qubits)
        : num_qubits_{num_qubits}, sv{num_qubits}, cuda_sv{num_qubits},
          cuda_sv_raw{cuda_sv.getData(), cuda_sv.getLength()} {
        cuda_sv.initSV();
    }
    SVDataGPURaw(std::size_t num_qubits,
                 const std::vector<std::complex<fp_t>> &cdata_input)
        : num_qubits_{num_qubits}, sv{cdata_input}, cuda_sv{cdata_input.data(),
                                                            cdata_input.size()},
          cuda_sv_raw{cuda_sv.getData(), cuda_sv.getLength()} {}

    ~SVDataGPURaw() {}
};