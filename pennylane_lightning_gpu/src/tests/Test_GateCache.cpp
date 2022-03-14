
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include <cuComplex.h> // cuDoubleComplex
#include <cuda.h>

#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace CUDA;

namespace {
namespace cuUtil = Pennylane::CUDA::Util;
} // namespace

TEMPLATE_TEST_CASE("CuGateCache", "[CuGateCache]", float, double) {
    using cp_t = std::complex<TestType>;
    using cp_dev_t = decltype(cuUtil::getCudaType(TestType{}));
    GateCache<TestType> gc(true);
    const std::size_t length = 4;

    SECTION("Hadamard gate") {
        std::vector<cp_t> H_transfer(length, cp_t{0, 0});
        auto H_host = gc.get_gate_host("Hadamard", 0.0);
        cudaMemcpy(reinterpret_cast<cp_dev_t *>(H_transfer.data()),
                   gc.get_gate_device_ptr("Hadamard", 0.0),
                   sizeof(cp_dev_t) * length, cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < length; i++) {
            CHECK(H_host[i].x == Approx(H_transfer[i].real()).epsilon(1e-7));
            CHECK(H_host[i].y == Approx(H_transfer[i].imag()).epsilon(1e-7));
        }
    }
}