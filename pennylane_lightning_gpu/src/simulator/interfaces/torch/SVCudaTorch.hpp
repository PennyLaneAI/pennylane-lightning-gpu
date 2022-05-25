#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuComplex.h>
#include <torch/extension.h>

#include "StateVectorCudaRaw.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::CUDA::Util;
using namespace Pennylane::CUDA;
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane {

/**
 * @brief Raw memory CUDA state-vector class using custateVec backed
 * gate-calls.
 *
 * @tparam Precision Floating-point precision type.
 */
template <class Precision>
class SVCudaTorch final : public StateVectorCudaRaw<Precision> {
  private:
    using BaseType = StateVectorCudaRaw<Precision>;

  public:
    using CFP_t = typename StateVectorCudaRaw<Precision>::CFP_t;
    using GateType = CFP_t *;

    SVCudaTorch() = delete;
    SVCudaTorch(torch::Tensor &tensor, cudaStream_t stream = 0)
        : StateVectorCudaRaw(Util::log2(tensor.numel()), stream) {
        data_ = reinterpret_cast<CFP_t *>(tensor.data_ptr());
    }

  private:
    CFP_t *data_;
};

}; // namespace Pennylane
