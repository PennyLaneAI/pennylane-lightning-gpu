#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointDiffGPU.hpp"
#include "StateVectorCudaManaged.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

using namespace Pennylane::CUDA;
using namespace Pennylane::Algorithms;

TEMPLATE_TEST_CASE("ObservablesGPU::NamedObsGPU", "[ObservablesGPU]", float,
                   double) {
    SECTION("NamedObsGPU<TestType> CTOR") {
        REQUIRE_FALSE(std::is_constructible<NamedObsGPU<TestType>>::value);
        REQUIRE(std::is_constructible<NamedObsGPU<TestType>, std::string,
                                      std::vector<size_t>,
                                      std::vector<TestType>>::value);
    }

    NamedObsGPU<TestType> obs1{"PauliX", {3}, {}};
    NamedObsGPU<TestType> obs2{"RX", {1}, {0.245}};
    NamedObsGPU<TestType> obs3{"UnsupportedObs", {2, 3}, {0.876, 1.5623}};

    SECTION("NamedObsGPU<TestType>::getWires") {
        CHECK(obs1.getWires() == std::vector<size_t>{3});
        CHECK(obs2.getWires() == std::vector<size_t>{1});
        CHECK(obs3.getWires() == std::vector<size_t>{2, 3});
    }
    SECTION("NamedObsGPU<TestType>::getObsName") {
        CHECK(obs1.getObsName() == "PauliX[3]");
        CHECK(obs2.getObsName() == "RX[1]");
        CHECK(obs3.getObsName() == "UnsupportedObs[2, 3]");
    }
    SECTION("NamedObsGPU<TestType>::applyInPlace") {
        StateVectorCudaManaged<TestType> sv(4);
        sv.initSV();

        std::vector<std::complex<TestType>> host_array(16, {0, 0});
        std::vector<std::complex<TestType>> res1 = {
            {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        std::vector<std::complex<TestType>> res2 = {
            {0, 0}, {0.99250625, 0}, {0, 0}, {0, 0}, {0, 0}, {0, -0.12219385},
            {0, 0}, {0, 0},          {0, 0}, {0, 0}, {0, 0}, {0, 0},
            {0, 0}, {0, 0},          {0, 0}, {0, 0}};

        obs1.applyInPlace(sv);
        sv.getDataBuffer().CopyGpuDataToHost(host_array.data(),
                                             host_array.size());
        for (std::size_t i = 0; i < host_array.size(); i++) {
            CHECK(host_array[i].real() == Approx(res1[i].real()));
            CHECK(host_array[i].imag() == Approx(res1[i].imag()));
        }

        obs2.applyInPlace(sv);
        sv.getDataBuffer().CopyGpuDataToHost(host_array.data(),
                                             host_array.size());
        for (std::size_t i = 0; i < host_array.size(); i++) {
            CHECK(host_array[i].real() == Approx(res2[i].real()));
            CHECK(host_array[i].imag() == Approx(res2[i].imag()));
        }
        REQUIRE_THROWS(obs3.applyInPlace(sv));
    }
}

TEMPLATE_TEST_CASE("ObservablesGPU::HermitianObsGPU", "[ObservablesGPU]", float,
                   double) {
    SECTION("HermitianObsGPU<TestType> CTOR") {
        REQUIRE_FALSE(std::is_constructible<HermitianObsGPU<TestType>>::value);
        REQUIRE(std::is_constructible<HermitianObsGPU<TestType>,
                                      std::vector<std::complex<TestType>>,
                                      std::vector<size_t>>::value);
    }
}

TEMPLATE_TEST_CASE("ObservablesGPU::HamiltonianGPU", "[ObservablesGPU]", float,
                   double) {
    SECTION("HamiltonianGPU<TestType> CTOR") {
        REQUIRE_FALSE(std::is_constructible<HamiltonianGPU<TestType>>::value);
        REQUIRE(std::is_constructible<
                HamiltonianGPU<TestType>, std::vector<TestType>,
                std::vector<std::shared_ptr<ObservableGPU<TestType>>>>::value);
    }
}

TEMPLATE_TEST_CASE("ObservablesGPU::TensorProdObsGPU", "[ObservablesGPU]",
                   float, double) {
    SECTION("TensorProdObsGPU<TestType> CTOR") {
        // Note: detecting empty parameter pack not trivially implementable
        REQUIRE(std::is_constructible<TensorProdObsGPU<TestType>,
                                      HermitianObsGPU<TestType>,
                                      NamedObsGPU<TestType>>::value);
    }
}
