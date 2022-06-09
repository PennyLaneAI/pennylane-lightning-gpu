
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "DataBuffer.hpp"

#include <cuComplex.h> // cuDoubleComplex
#include <cuda.h>

#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace CUDA;

namespace {
namespace cuUtil = Pennylane::CUDA::Util;
} // namespace


/**
 * @brief Tests the constructability of the DataBuffer class.
 *
 */
TEMPLATE_TEST_CASE("DataBuffer::DataBuffer",
                   "[DataBuffer]", char, int, unsigned int, long, float, double, float2, double2) {
    SECTION("DataBuffer<GPUDataT>{std::size_t, int, cudaStream_t, bool}"){
        REQUIRE(std::is_constructible<DataBuffer<TestType, int>,
                                      std::size_t, int, cudaStream_t, bool>::value);
    }
    SECTION("DataBuffer<GPUDataT>{std::size_t, const DevTag<int>&, bool}"){
        REQUIRE(std::is_constructible<DataBuffer<TestType, int>,
                                      std::size_t, const DevTag<int>&, bool>::value);
    }
    SECTION("DataBuffer<GPUDataT>{std::size_t, DevTag<int>&&, bool}"){
        REQUIRE(std::is_constructible<DataBuffer<TestType, int>,
                                      std::size_t, DevTag<int>&&, bool>::value);
    }
    SECTION("DataBuffer<GPUDataT>=default non-constructable"){
        REQUIRE_FALSE(std::is_default_constructible<DataBuffer<TestType, int>>::value);
    }
}
