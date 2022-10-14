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

    SECTION("NamedObsGPU<TestType> binary ops") {
        CHECK(obs1 == NamedObsGPU<TestType>{"PauliX", {3}, {}});
        CHECK(obs1 != obs2);
    }
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
        CHECK_THROWS(obs3.applyInPlace(sv));
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

    std::vector<std::complex<TestType>> hermitian_h{{0.7071067811865475, 0},
                                                    {0.7071067811865475, 0},
                                                    {0.7071067811865475, 0},
                                                    {-0.7071067811865475, 0}};
    std::vector<std::complex<TestType>> hermitian_ry{{std::cos(0.1234), 0},
                                                     {-std::sin(0.1234), 0},
                                                     {std::sin(0.1234), 0},
                                                     {std::cos(0.1234), 0}};

    HermitianObsGPU<TestType> obs1{hermitian_h, {0}};
    HermitianObsGPU<TestType> obs2{hermitian_ry, {2}};

    SECTION("HermitianObsGPU<TestType> binary ops") {
        CHECK(obs1 == HermitianObsGPU<TestType>{hermitian_h, {0}});
        CHECK(obs1 != obs2);
    }
    SECTION("HermitianObsGPU<TestType>::getWires") {
        CHECK(obs1.getWires() == std::vector<size_t>{0});
        CHECK(obs2.getWires() == std::vector<size_t>{2});
    }

    SECTION("HermitianObsGPU<TestType>::applyInPlace") {
        StateVectorCudaManaged<TestType> sv(4);
        sv.initSV();

        std::vector<std::complex<TestType>> host_array(16, {0, 0});
        std::vector<std::complex<TestType>> res1 = {{0.7071067811865475, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0.7071067811865475, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0},
                                                    {0, 0}};
        std::vector<std::complex<TestType>> res2 = {
            {0.30734708, 0}, {0, 0}, {0.39438277, 0}, {0, 0},
            {0.30734708, 0}, {0, 0}, {0.39438277, 0}, {0, 0},
            {0.30734708, 0}, {0, 0}, {0.39438277, 0}, {0, 0},
            {0.30734708, 0}, {0, 0}, {0.39438277, 0}, {0, 0}};

        obs1.applyInPlace(sv);
        sv.getDataBuffer().CopyGpuDataToHost(host_array.data(),
                                             host_array.size());
        for (std::size_t i = 0; i < host_array.size(); i++) {
            CHECK(host_array[i].real() == Approx(res1[i].real()));
            CHECK(host_array[i].imag() == Approx(res1[i].imag()));
        }
        sv.applyHadamard({1}, false);
        sv.applyHadamard({2}, false);
        obs2.applyInPlace(sv);

        sv.getDataBuffer().CopyGpuDataToHost(host_array.data(),
                                             host_array.size());
        for (std::size_t i = 0; i < host_array.size(); i++) {
            CHECK(host_array[i].real() == Approx(res2[i].real()));
            CHECK(host_array[i].imag() == Approx(res2[i].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("ObservablesGPU::TensorProdObsGPU", "[ObservablesGPU]",
                   float, double) {
    SECTION("TensorProdObsGPU<TestType> CTOR") {
        // Note: Empty parameter pack will be constructable.
        REQUIRE(std::is_constructible<TensorProdObsGPU<TestType>>::value);
        REQUIRE(std::is_constructible<
                TensorProdObsGPU<TestType>,
                std::shared_ptr<NamedObsGPU<TestType>>,
                std::shared_ptr<NamedObsGPU<TestType>>>::value);
        REQUIRE(std::is_constructible<
                TensorProdObsGPU<TestType>,
                std::shared_ptr<NamedObsGPU<TestType>>,
                std::shared_ptr<HermitianObsGPU<TestType>>>::value);
        REQUIRE(std::is_constructible<
                TensorProdObsGPU<TestType>,
                std::shared_ptr<HermitianObsGPU<TestType>>,
                std::shared_ptr<HermitianObsGPU<TestType>>>::value);
    }

    std::vector<std::complex<TestType>> hermitian_h{{0.7071067811865475, 0},
                                                    {0.7071067811865475, 0},
                                                    {0.7071067811865475, 0},
                                                    {-0.7071067811865475, 0}};
    std::vector<std::complex<TestType>> hermitian_ry{{std::cos(0.1234), 0},
                                                     {-std::sin(0.1234), 0},
                                                     {std::sin(0.1234), 0},
                                                     {std::cos(0.1234), 0}};

    std::shared_ptr<HermitianObsGPU<TestType>> obs1(
        new HermitianObsGPU<TestType>(hermitian_h, {0}));
    std::shared_ptr<HermitianObsGPU<TestType>> obs2(
        new HermitianObsGPU<TestType>(hermitian_ry, {2}));
    std::shared_ptr<NamedObsGPU<TestType>> obs3(
        new NamedObsGPU<TestType>("PauliX", {3}, {}));
    std::shared_ptr<NamedObsGPU<TestType>> obs4(
        new NamedObsGPU<TestType>("RX", {1}, {0.245}));

    TensorProdObsGPU<TestType> tp_obs1{obs1, obs2};
    TensorProdObsGPU<TestType> tp_obs2{obs1, obs3};
    TensorProdObsGPU<TestType> tp_obs3{obs3, obs4};

    SECTION("TensorProdObsGPU<TestType> binary ops") {
        CHECK(tp_obs1 == TensorProdObsGPU<TestType>{obs1, obs2});
        CHECK(tp_obs1 != tp_obs2);
    }
    SECTION("TensorProdObsGPU<TestType>::getWires") {
        CHECK(tp_obs1.getWires() == std::vector<size_t>{0, 2});
        CHECK(tp_obs2.getWires() == std::vector<size_t>{0, 3});
        CHECK(tp_obs3.getWires() == std::vector<size_t>{1, 3});
        CHECK_THROWS(TensorProdObsGPU<TestType>{obs1, obs1});
    }

    SECTION("TensorProdObsGPU<TestType>::applyInPlace") {
        StateVectorCudaManaged<TestType> sv(4);
        sv.initSV();
        sv.applyHadamard({1}, false);
        sv.applyHadamard({2}, false);

        std::vector<std::complex<TestType>> host_array(16, {0, 0});
        std::vector<std::complex<TestType>> res1 = {
            {0.30734708, 0}, {0, 0}, {0.39438277, 0}, {0, 0},
            {0.30734708, 0}, {0, 0}, {0.39438277, 0}, {0, 0},
            {0.30734708, 0}, {0, 0}, {0.39438277, 0}, {0, 0},
            {0.30734708, 0}, {0, 0}, {0.39438277, 0}, {0, 0}};

        tp_obs1.applyInPlace(sv);
        sv.getDataBuffer().CopyGpuDataToHost(host_array.data(),
                                             host_array.size());
        for (std::size_t i = 0; i < host_array.size(); i++) {
            CHECK(host_array[i].real() == Approx(res1[i].real()));
            CHECK(host_array[i].imag() == Approx(res1[i].imag()));
        }

        std::vector<std::complex<TestType>> res2 = {
            {0, 0}, {0.3050439, -0.03755592},
            {0, 0}, {0.39142737, -0.04819115},
            {0, 0}, {0.3050439, -0.03755592},
            {0, 0}, {0.39142737, -0.04819115},
            {0, 0}, {0.3050439, -0.03755592},
            {0, 0}, {0.39142737, -0.04819115},
            {0, 0}, {0.3050439, -0.03755592},
            {0, 0}, {0.39142737, -0.04819115}};

        tp_obs3.applyInPlace(sv);
        sv.getDataBuffer().CopyGpuDataToHost(host_array.data(),
                                             host_array.size());
        for (std::size_t i = 0; i < host_array.size(); i++) {
            CHECK(host_array[i].real() == Approx(res2[i].real()));
            CHECK(host_array[i].imag() == Approx(res2[i].imag()));
        }
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

    std::vector<std::complex<TestType>> hermitian_h{{0.7071067811865475, 0},
                                                    {0.7071067811865475, 0},
                                                    {0.7071067811865475, 0},
                                                    {-0.7071067811865475, 0}};

    std::shared_ptr<HermitianObsGPU<TestType>> obs1(
        new HermitianObsGPU<TestType>(hermitian_h, {0}));
    std::shared_ptr<NamedObsGPU<TestType>> obs2(
        new NamedObsGPU<TestType>("PauliX", {2}, {}));
    std::shared_ptr<NamedObsGPU<TestType>> obs3(
        new NamedObsGPU<TestType>("PauliX", {3}, {}));

    std::shared_ptr<TensorProdObsGPU<TestType>> tp_obs1(
        new TensorProdObsGPU<TestType>(obs1, obs2));
    std::shared_ptr<TensorProdObsGPU<TestType>> tp_obs2(
        new TensorProdObsGPU<TestType>(obs2, obs3));

    HamiltonianGPU<TestType> ham_1{
        std::vector<TestType>{0.165, 0.13, 0.5423},
        std::vector<std::shared_ptr<ObservableGPU<TestType>>>{obs1, obs2,
                                                              obs2}};
    HamiltonianGPU<TestType> ham_2{
        std::vector<TestType>{0.8545, 0.3222},
        std::vector<std::shared_ptr<ObservableGPU<TestType>>>{tp_obs1,
                                                              tp_obs2}};

    SECTION("HamiltonianGPU<TestType> binary ops") {
        CHECK(ham_1 ==
              HamiltonianGPU<TestType>{
                  std::vector<TestType>{0.165, 0.13, 0.5423},
                  std::vector<std::shared_ptr<ObservableGPU<TestType>>>{
                      obs1, obs2, obs2}});
        CHECK(ham_1 != ham_2);
    }
    SECTION("HamiltonianGPU<TestType>::getWires") {
        CHECK(ham_1.getWires() == std::vector<size_t>{0, 2});
        CHECK(ham_2.getWires() == std::vector<size_t>{0, 2, 3});
    }
    SECTION("HamiltonianGPU<TestType>::obsName") {
        std::ostringstream res1, res2;
        res1 << "Hamiltonian: { 'coeffs' : [0.165, 0.13, 0.5423], "
                "'observables' : [Hermitian"
             << MatrixHasher()(hermitian_h) << ", PauliX[2], PauliX[2]]}";
        res2 << "Hamiltonian: { 'coeffs' : [0.8545, 0.3222], 'observables' : "
                "[Hermitian"
             << MatrixHasher()(hermitian_h)
             << " @ PauliX[2], PauliX[2] @ PauliX[3]]}";

        CHECK(ham_1.getObsName() == res1.str());
        CHECK(ham_2.getObsName() == res2.str());
    }

    SECTION("HamiltonianGPU<TestType>::applyInPlace") {
        StateVectorCudaManaged<TestType> sv(4);
        sv.initSV();
        sv.applyHadamard({0}, false);
        sv.applyHadamard({1}, false);
        sv.applyHadamard({2}, false);

        std::vector<std::complex<TestType>> host_array(16, {0, 0});
        std::vector<std::complex<TestType>> res1 = {
            {0.32019394, 0}, {0, 0}, {0.32019394, 0}, {0, 0},
            {0.32019394, 0}, {0, 0}, {0.32019394, 0}, {0, 0},
            {0.23769394, 0}, {0, 0}, {0.23769394, 0}, {0, 0},
            {0.23769394, 0}, {0, 0}, {0.23769394, 0}, {0, 0}};

        ham_1.applyInPlace(sv);
        sv.getDataBuffer().CopyGpuDataToHost(host_array.data(),
                                             host_array.size());
        for (std::size_t i = 0; i < host_array.size(); i++) {
            CHECK(host_array[i].real() == Approx(res1[i].real()));
            CHECK(host_array[i].imag() == Approx(res1[i].imag()));
        }

        std::vector<std::complex<TestType>> res2 = {
            {0.33708855, 0}, {0.10316649, 0}, {0.33708855, 0}, {0.10316649, 0},
            {0.33708855, 0}, {0.10316649, 0}, {0.33708855, 0}, {0.10316649, 0},

            {0.04984838, 0}, {0.07658499, 0}, {0.04984838, 0}, {0.07658499, 0},
            {0.04984838, 0}, {0.07658499, 0}, {0.04984838, 0}, {0.07658499, 0}};

        ham_2.applyInPlace(sv);
        sv.getDataBuffer().CopyGpuDataToHost(host_array.data(),
                                             host_array.size());
        for (std::size_t i = 0; i < host_array.size(); i++) {
            CHECK(host_array[i].real() == Approx(res2[i].real()));
            CHECK(host_array[i].imag() == Approx(res2[i].imag()));
        }
    }
}
