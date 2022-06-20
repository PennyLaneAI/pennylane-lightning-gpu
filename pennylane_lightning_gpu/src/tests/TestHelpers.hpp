#include <complex>
#include <vector>

#include "AdjointDiffGPU.hpp"
#include "StateVectorCudaManaged.hpp"
#include "StateVectorManagedCPU.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

namespace Pennylane {
template <class T, class Alloc = std::allocator<T>> struct PLApprox {
    const std::vector<T, Alloc> &comp_;

    explicit PLApprox(const std::vector<T, Alloc> &comp) : comp_{comp} {}

    Pennylane::Util::remove_complex_t<T> margin_{};
    Pennylane::Util::remove_complex_t<T> epsilon_ =
        std::numeric_limits<float>::epsilon() * 100;

    template <class AllocA>
    [[nodiscard]] bool compare(const std::vector<T, AllocA> &lhs) const {
        if (lhs.size() != comp_.size()) {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); i++) {
            if constexpr (Pennylane::Util::is_complex_v<T>) {
                if (lhs[i].real() != Approx(comp_[i].real())
                                         .epsilon(epsilon_)
                                         .margin(margin_) ||
                    lhs[i].imag() != Approx(comp_[i].imag())
                                         .epsilon(epsilon_)
                                         .margin(margin_)) {
                    return false;
                }
            } else {
                if (lhs[i] !=
                    Approx(comp_[i]).epsilon(epsilon_).margin(margin_)) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] std::string describe() const {
        std::ostringstream ss;
        ss << "is Approx to {";
        for (const auto &elt : comp_) {
            ss << elt << ", ";
        }
        ss << "}" << std::endl;
        return ss.str();
    }

    PLApprox &epsilon(Pennylane::Util::remove_complex_t<T> eps) {
        epsilon_ = eps;
        return *this;
    }
    PLApprox &margin(Pennylane::Util::remove_complex_t<T> m) {
        margin_ = m;
        return *this;
    }
};

/**
 * @brief Simple helper for PLApprox for the cases when the class template
 * deduction does not work well.
 */
template <typename T, class Alloc>
PLApprox<T, Alloc> approx(const std::vector<T, Alloc> &vec) {
    return PLApprox<T, Alloc>(vec);
}

template <typename T, class Alloc>
std::ostream &operator<<(std::ostream &os, const PLApprox<T, Alloc> &approx) {
    os << approx.describe();
    return os;
}
template <class T, class AllocA, class AllocB>
bool operator==(const std::vector<T, AllocA> &lhs,
                const PLApprox<T, AllocB> &rhs) {
    return rhs.compare(lhs);
}
template <class T, class AllocA, class AllocB>
bool operator!=(const std::vector<T, AllocA> &lhs,
                const PLApprox<T, AllocB> &rhs) {
    return !rhs.compare(lhs);
}
}; // namespace Pennylane

namespace {
using namespace Pennylane;
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
    StateVectorManagedCPU<fp_t> sv;
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
    SVDataGPU(
        std::size_t num_qubits,
        const std::vector<std::complex<fp_t>,
                          Pennylane::Util::AlignedAllocator<std::complex<fp_t>>>
            &cdata_input)
        : num_qubits_{num_qubits}, sv{cdata_input}, cuda_sv{
                                                        cdata_input.data(),
                                                        cdata_input.size()} {}

    ~SVDataGPU() {}
};