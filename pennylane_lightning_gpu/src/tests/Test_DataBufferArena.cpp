
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DataBufferArena.hpp"
#include "DevTag.hpp"
#include "DevicePool.hpp"

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
TEMPLATE_TEST_CASE("DataBufferArena::DataBufferArena", "[DataBufferArena]",
                   char, int, unsigned int, long, float, double, float2,
                   double2) {
    SECTION("DataBufferArena<GPUDataT>{std:vector<std::size_t>, int, "
            "cudaStream_t, bool}") {
        REQUIRE(std::is_constructible<DataBufferArena<TestType, int>,
                                      std::vector<std::size_t>, int,
                                      cudaStream_t, bool>::value);
    }
    SECTION("DataBufferArena<GPUDataT>{const std:vector<std::size_t>&, const "
            "DevTag<int>&, bool}") {
        REQUIRE(std::is_constructible<DataBufferArena<TestType, int>,
                                      const std::vector<std::size_t> &,
                                      const DevTag<int> &, bool>::value);
    }

    SECTION("DataBufferArena<GPUDataT>{const std:vector<std::size_t>&, "
            "DevTag<int>&&, bool}") {
        REQUIRE(std::is_constructible<DataBufferArena<TestType, int>,
                                      const std::vector<std::size_t> &,
                                      DevTag<int> &&, bool>::value);
    }
    SECTION("DataBufferArena<GPUDataT>=default non-constructable") {
        REQUIRE_FALSE(std::is_default_constructible<
                      DataBufferArena<TestType, int>>::value);
    }
}

TEMPLATE_TEST_CASE("DataBufferArena::memory allocation", "[DataBufferArena]",
                   float, double) {
    SECTION("Allocate buffer memory = true") {
        DataBufferArena<TestType, int> data_buffer1{{8}, 0, 0, true};
        CHECK(data_buffer1.getData() != nullptr);
        CHECK(data_buffer1.getLength() == 8);
        CHECK(data_buffer1.getStream() == 0);
        CHECK(data_buffer1.getDevice() == 0);
    }
    SECTION("Allocate buffer memory = false") {
        DataBufferArena<TestType, int> data_buffer1{{7}, 0, 0, false};
        CHECK(data_buffer1.getData() == nullptr);
        CHECK(data_buffer1.getLength() == 7);
        CHECK(data_buffer1.getStream() == 0);
        CHECK(data_buffer1.getDevice() == 0);
    }
}
/*
TEMPLATE_TEST_CASE("Data locality and movement", "[DataBufferArena]", float,
                   double) {
    SECTION("Single gpu movement") {
        DataBufferArena<TestType, int> data_buffer1{{6}, 0, 0, true};
        std::vector<TestType> host_data_in(6, 1);
        std::vector<TestType> host_data_out(6, 0);
        data_buffer1.CopyHostDataToGpu(host_data_in.data(), host_data_in.size(),
                                       false);
        DataBufferArena<TestType, int> data_buffer2(
            data_buffer1.getBufferLengths(), data_buffer1.getDevTag(), true);
        data_buffer2.CopyGpuDataToGpu(data_buffer1, false);
        data_buffer2.CopyGpuDataToHost(host_data_out.data(), 6, false);
        CHECK(host_data_in == host_data_out);
        CHECK(data_buffer1.getLength() == data_buffer2.getLength());
        CHECK(data_buffer1.getData() !=
              data_buffer2.getData()); // Ptrs should not refer to same block
    }
    if (DevicePool<int>::getTotalDevices() > 1) {
        SECTION("Multi-GPU copy") {
            DevicePool<int> dev_pool;
            std::vector<int> ids;
            std::vector<DevTag<int>> tags;
            std::vector<std::unique_ptr<DataBufferArena<TestType, int>>>
                buffers;
            for (std::size_t i = 0; i < dev_pool.getTotalDevices(); i++) {
                ids.push_back(dev_pool.acquireDevice());
                tags.push_back({ids.back(), 0U});
                buffers.emplace_back(
                    std::make_unique<DataBufferArena<TestType, int>>(
                        6, tags.back(), true));
            }

            std::vector<TestType> host_data_in(6, 1);
            std::vector<TestType> host_data_out(6, 0);
            buffers[0]->CopyHostDataToGpu(host_data_in.data(),
                                          host_data_in.size(), false);
            for (std::size_t i = 1; i < dev_pool.getTotalDevices(); i++) {
                buffers[i]->CopyGpuDataToGpu(*buffers[i - 1], false);
            }
            buffers.back()->CopyGpuDataToHost(host_data_out.data(), 6, false);
            CHECK(host_data_in == host_data_out);
            for (auto &id : ids) {
                dev_pool.releaseDevice(id);
            }
        }
    }
}

*/