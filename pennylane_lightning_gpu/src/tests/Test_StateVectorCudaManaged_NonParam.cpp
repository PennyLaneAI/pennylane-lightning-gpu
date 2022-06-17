
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVectorCudaManaged.hpp"
#include "StateVectorRawCPU.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace CUDA;

namespace {
namespace cuUtil = Pennylane::CUDA::Util;
} // namespace

/**
 * @brief Tests the constructability of the StateVectorCudaManaged class.
 *
 */
TEMPLATE_TEST_CASE("StateVectorCudaManaged::StateVectorCudaManaged",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    SECTION("StateVectorCudaManaged<TestType> {std::complex<TestType>, "
            "std::size_t}") {
        REQUIRE(std::is_constructible<StateVectorCudaManaged<TestType>,
                                      std::complex<TestType> *,
                                      std::size_t>::value);
    }
    SECTION("StateVectorCudaManaged<TestType> cross types") {
        if constexpr (!std::is_same_v<TestType, double>) {
            REQUIRE_FALSE(
                std::is_constructible<StateVectorCudaManaged<TestType>,
                                      std::complex<double> *,
                                      std::size_t>::value);
            REQUIRE_FALSE(std::is_constructible<StateVectorCudaManaged<double>,
                                                std::complex<TestType> *,
                                                std::size_t>::value);
        } else if constexpr (!std::is_same_v<TestType, float>) {
            REQUIRE_FALSE(
                std::is_constructible<StateVectorCudaManaged<TestType>,
                                      std::complex<float> *,
                                      std::size_t>::value);
            REQUIRE_FALSE(std::is_constructible<StateVectorCudaManaged<float>,
                                                std::complex<TestType> *,
                                                std::size_t>::value);
        }
    }
    SECTION("StateVectorCudaManaged<TestType> transfers") {
        using cp_t = std::complex<TestType>;
        const std::size_t num_qubits = 3;
        const std::vector<cp_t> init_state{{1, 0}, {0, 0}, {0, 0}, {0, 0},
                                           {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        SECTION("GPU <-> host data: std::complex") {
            SVDataGPU<TestType> svdat{num_qubits};
            std::vector<cp_t> out_data(Pennylane::Util::exp2(num_qubits),
                                       {0.5, 0.5});

            CHECK(svdat.sv.getDataVector() == Pennylane::approx(init_state));
            svdat.cuda_sv.CopyGpuDataToHost(out_data.data(), out_data.size());
            CHECK(out_data == init_state);

            svdat.sv.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                     {{0}, {1}, {2}}, {false, false, false});
            svdat.cuda_sv.CopyHostDataToGpu(svdat.sv);
            svdat.cuda_sv.CopyGpuDataToHost(out_data.data(), out_data.size());
            CHECK(out_data == Pennylane::approx(svdat.sv.getDataVector()));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyHadamard",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SECTION("Apply directly") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat{num_qubits};

            CHECK(svdat.sv.getDataVector()[0] == cp_t{1, 0});
            svdat.cuda_sv.applyHadamard({index}, false);
            CAPTURE(svdat.sv.getDataVector());
            svdat.cuda_sv.CopyGpuDataToHost(svdat.sv.getData(),
                                            svdat.sv.getLength());
            CAPTURE(svdat.sv.getDataVector());

            cp_t expected(1 / std::sqrt(2), 0);
            CHECK(expected.real() ==
                  Approx(svdat.sv.getDataVector()[0].real()));
            CHECK(expected.imag() ==
                  Approx(svdat.sv.getDataVector()[0].imag()));

            CHECK(
                expected.real() ==
                Approx(
                    svdat.sv
                        .getDataVector()[0b1 << (svdat.num_qubits_ - index - 1)]
                        .real()));
            CHECK(
                expected.imag() ==
                Approx(
                    svdat.sv
                        .getDataVector()[0b1 << (svdat.num_qubits_ - index - 1)]
                        .imag()));
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat(num_qubits);

            CHECK(svdat.sv.getDataVector()[0] == cp_t{1, 0});
            svdat.cuda_sv.applyOperation("Hadamard", {index}, false);
            svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
            cp_t expected(1.0 / std::sqrt(2), 0);

            CHECK(expected.real() ==
                  Approx(svdat.sv.getDataVector()[0].real()));
            CHECK(expected.imag() ==
                  Approx(svdat.sv.getDataVector()[0].imag()));

            CHECK(
                expected.real() ==
                Approx(
                    svdat.sv
                        .getDataVector()[0b1 << (svdat.num_qubits_ - index - 1)]
                        .real()));
            CHECK(
                expected.imag() ==
                Approx(
                    svdat.sv
                        .getDataVector()[0b1 << (svdat.num_qubits_ - index - 1)]
                        .imag()));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyPauliX",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    // using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SECTION("Apply directly") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat{num_qubits};
            CHECK(svdat.sv.getDataVector()[0] ==
                  cuUtil::ONE<std::complex<TestType>>());
            svdat.cuda_sv.applyPauliX({index}, false);
            svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
            CHECK(svdat.sv.getDataVector()[0] ==
                  cuUtil::ZERO<std::complex<TestType>>());
            CHECK(
                svdat.sv
                    .getDataVector()[0b1 << (svdat.num_qubits_ - index - 1)] ==
                cuUtil::ONE<std::complex<TestType>>());
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat{num_qubits};
            CHECK(svdat.sv.getDataVector()[0] ==
                  cuUtil::ONE<std::complex<TestType>>());
            svdat.cuda_sv.applyOperation("PauliX", {index}, false);
            svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
            CHECK(svdat.sv.getDataVector()[0] ==
                  cuUtil::ZERO<std::complex<TestType>>());
            CHECK(
                svdat.sv
                    .getDataVector()[0b1 << (svdat.num_qubits_ - index - 1)] ==
                cuUtil::ONE<std::complex<TestType>>());
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyPauliY",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});

    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);

    const cp_t p = cuUtil::ConstMult(
        std::complex<TestType>(0.5, 0.0),
        cuUtil::ConstMult(cuUtil::INVSQRT2<std::complex<TestType>>(),
                          cuUtil::IMAG<std::complex<TestType>>()));
    const cp_t m = cuUtil::ConstMult(std::complex<TestType>(-1, 0), p);

    const std::vector<std::vector<cp_t>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_direct{num_qubits, init_state};

            CHECK(svdat_direct.sv.getDataVector() == init_state);
            svdat_direct.cuda_sv.applyPauliY({index}, false);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
            CHECK(svdat_dispatch.sv.getDataVector() == init_state);
            svdat_dispatch.cuda_sv.applyOperation("PauliY", {index}, false);
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyPauliZ",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});

    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
    const cp_t p(static_cast<TestType>(0.5) *
                 cuUtil::INVSQRT2<std::complex<TestType>>());
    const cp_t m(cuUtil::ConstMult(cp_t{-1.0, 0.0}, p));

    const std::vector<std::vector<cp_t>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
            CHECK(svdat_direct.sv.getDataVector() == init_state);
            svdat_direct.cuda_sv.applyPauliZ({index}, false);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
            CHECK(svdat_dispatch.sv.getDataVector() == init_state);
            svdat_dispatch.cuda_sv.applyOperation("PauliZ", {index}, false);
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyS",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const cp_t r(std::complex<TestType>(0.5, 0.0) *
                 cuUtil::INVSQRT2<std::complex<TestType>>());
    const cp_t i(cuUtil::ConstMult(r, cuUtil::IMAG<std::complex<TestType>>()));

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
            CHECK(svdat_direct.sv.getDataVector() == init_state);
            svdat_direct.cuda_sv.applyS({index}, false);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
            CHECK(svdat_dispatch.sv.getDataVector() == init_state);
            svdat_dispatch.cuda_sv.applyOperation("S", {index}, false);
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyT",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});

    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
    cp_t r(1.0 / (2.0 * std::sqrt(2)), 0);
    cp_t i(1.0 / 4, 1.0 / 4);

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
            CHECK(svdat_direct.sv.getDataVector() == init_state);
            svdat_direct.cuda_sv.applyT({index}, false);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CAPTURE(svdat_direct.sv.getDataVector());
            CAPTURE(expected_results[index]);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
            CHECK(svdat_dispatch.sv.getDataVector() == init_state);
            svdat_dispatch.cuda_sv.applyOperation("T", {index}, false);
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyCNOT",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    // using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    // Test using |+00> state to generate 3-qubit GHZ state
    svdat.sv.applyOperation("Hadamard", {0});
    const auto init_state = svdat.sv.getDataVector();

    SECTION("Apply directly") {
        SVDataGPU<TestType> svdat_direct{num_qubits, init_state};

        for (std::size_t index = 1; index < num_qubits; index++) {
            svdat_direct.cuda_sv.applyCNOT({index - 1, index}, false);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
        }
        CHECK(svdat_direct.sv.getDataVector().front() ==
              cuUtil::INVSQRT2<std::complex<TestType>>());
        CHECK(svdat_direct.sv.getDataVector().back() ==
              cuUtil::INVSQRT2<std::complex<TestType>>());
    }

    SECTION("Apply using dispatcher") {
        SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};

        for (std::size_t index = 1; index < num_qubits; index++) {
            svdat_dispatch.cuda_sv.applyOperation("CNOT", {index - 1, index},
                                                  false);
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
        }
        CHECK(svdat_dispatch.sv.getDataVector().front() ==
              cuUtil::INVSQRT2<std::complex<TestType>>());
        CHECK(svdat_dispatch.sv.getDataVector().back() ==
              cuUtil::INVSQRT2<std::complex<TestType>>());
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applySWAP",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                 {false, false});
    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
    const auto init_state = svdat.sv.getDataVector();

    SECTION("Apply directly") {
        CHECK(svdat.sv.getDataVector() ==
              Pennylane::approx(
                  std::vector<cp_t>{cuUtil::ZERO<std::complex<TestType>>(),
                                    cuUtil::ZERO<std::complex<TestType>>(),
                                    cuUtil::INVSQRT2<std::complex<TestType>>(),
                                    cuUtil::ZERO<std::complex<TestType>>(),
                                    cuUtil::ZERO<std::complex<TestType>>(),
                                    cuUtil::ZERO<std::complex<TestType>>(),
                                    cuUtil::INVSQRT2<std::complex<TestType>>(),
                                    cuUtil::ZERO<std::complex<TestType>>()}));

        SECTION("SWAP0,1 |+10> -> |1+0>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat01(num_qubits, init_state);
            SVDataGPU<TestType> svdat10(num_qubits, init_state);

            svdat01.cuda_sv.applySWAP({0, 1}, false);
            svdat10.cuda_sv.applySWAP({1, 0}, false);
            svdat01.cuda_sv.CopyGpuDataToHost(svdat01.sv);
            svdat10.cuda_sv.CopyGpuDataToHost(svdat10.sv);

            CHECK(svdat01.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat10.sv.getDataVector() == Pennylane::approx(expected));
        }

        SECTION("SWAP0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat02{num_qubits, init_state};
            SVDataGPU<TestType> svdat20{num_qubits, init_state};

            svdat02.cuda_sv.applySWAP({0, 2}, false);
            svdat20.cuda_sv.applySWAP({2, 0}, false);

            svdat02.cuda_sv.CopyGpuDataToHost(svdat02.sv);
            svdat20.cuda_sv.CopyGpuDataToHost(svdat20.sv);

            CHECK(svdat02.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat20.sv.getDataVector() == Pennylane::approx(expected));
        }
        SECTION("SWAP1,2 |+10> -> |+01>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat12{num_qubits, init_state};
            SVDataGPU<TestType> svdat21{num_qubits, init_state};

            svdat12.cuda_sv.applySWAP({1, 2}, false);
            svdat21.cuda_sv.applySWAP({2, 1}, false);

            svdat12.cuda_sv.CopyGpuDataToHost(svdat12.sv);
            svdat21.cuda_sv.CopyGpuDataToHost(svdat21.sv);
            ;

            CHECK(svdat12.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat21.sv.getDataVector() == Pennylane::approx(expected));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("SWAP0,1 |+10> -> |1+0>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat01{num_qubits, init_state};
            SVDataGPU<TestType> svdat10{num_qubits, init_state};

            svdat01.cuda_sv.applyOperation("SWAP", {0, 1});
            svdat10.cuda_sv.applyOperation("SWAP", {1, 0});

            svdat01.cuda_sv.CopyGpuDataToHost(svdat01.sv);
            svdat10.cuda_sv.CopyGpuDataToHost(svdat10.sv);
            ;

            CHECK(svdat01.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat10.sv.getDataVector() == Pennylane::approx(expected));
        }

        SECTION("SWAP0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat02{num_qubits, init_state};
            SVDataGPU<TestType> svdat20{num_qubits, init_state};

            svdat02.cuda_sv.applyOperation("SWAP", {0, 2});
            svdat20.cuda_sv.applyOperation("SWAP", {2, 0});

            svdat02.cuda_sv.CopyGpuDataToHost(svdat02.sv);
            svdat20.cuda_sv.CopyGpuDataToHost(svdat20.sv);

            CHECK(svdat02.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat20.sv.getDataVector() == Pennylane::approx(expected));
        }
        SECTION("SWAP1,2 |+10> -> |+01>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat12{num_qubits, init_state};
            SVDataGPU<TestType> svdat21{num_qubits, init_state};

            svdat12.cuda_sv.applyOperation("SWAP", {1, 2});
            svdat21.cuda_sv.applyOperation("SWAP", {2, 1});

            svdat12.cuda_sv.CopyGpuDataToHost(svdat12.sv);
            svdat21.cuda_sv.CopyGpuDataToHost(svdat21.sv);

            CHECK(svdat12.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat21.sv.getDataVector() == Pennylane::approx(expected));
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyCZ",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                 {false, false});
    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
    const auto init_state = svdat.sv.getDataVector();

    SECTION("Apply directly") {
        CHECK(svdat.sv.getDataVector() ==
              Pennylane::approx(
                  std::vector<cp_t>{cuUtil::ZERO<std::complex<TestType>>(),
                                    cuUtil::ZERO<std::complex<TestType>>(),
                                    std::complex<TestType>(1.0 / sqrt(2), 0),
                                    cuUtil::ZERO<std::complex<TestType>>(),
                                    cuUtil::ZERO<std::complex<TestType>>(),
                                    cuUtil::ZERO<std::complex<TestType>>(),
                                    std::complex<TestType>(1.0 / sqrt(2), 0),
                                    cuUtil::ZERO<std::complex<TestType>>()}));

        SECTION("CZ0,1 |+10> -> |-10>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(-1 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat01{num_qubits, init_state};
            SVDataGPU<TestType> svdat10{num_qubits, init_state};

            svdat01.cuda_sv.applyCZ({0, 1}, false);
            svdat10.cuda_sv.applyCZ({1, 0}, false);

            svdat01.cuda_sv.CopyGpuDataToHost(svdat01.sv);
            svdat10.cuda_sv.CopyGpuDataToHost(svdat10.sv);

            CHECK(svdat01.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat10.sv.getDataVector() == Pennylane::approx(expected));
        }

        SECTION("CZ0,2 |+10> -> |+10>") {
            auto &&expected = init_state;

            SVDataGPU<TestType> svdat02{num_qubits, init_state};
            SVDataGPU<TestType> svdat20{num_qubits, init_state};

            svdat02.cuda_sv.applyCZ({0, 2}, false);
            svdat20.cuda_sv.applyCZ({2, 0}, false);

            svdat02.cuda_sv.CopyGpuDataToHost(svdat02.sv);
            svdat20.cuda_sv.CopyGpuDataToHost(svdat20.sv);

            CHECK(svdat02.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat20.sv.getDataVector() == Pennylane::approx(expected));
        }
        SECTION("CZ1,2 |+10> -> |+10>") {
            auto &&expected = init_state;

            SVDataGPU<TestType> svdat12{num_qubits, init_state};
            SVDataGPU<TestType> svdat21{num_qubits, init_state};

            svdat12.cuda_sv.applyCZ({1, 2}, false);
            svdat21.cuda_sv.applyCZ({2, 1}, false);

            svdat12.cuda_sv.CopyGpuDataToHost(svdat12.sv);
            svdat21.cuda_sv.CopyGpuDataToHost(svdat21.sv);

            CHECK(svdat12.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat21.sv.getDataVector() == Pennylane::approx(expected));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CZ0,1 |+10> -> |1+0>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(-1 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat01{num_qubits, init_state};
            SVDataGPU<TestType> svdat10{num_qubits, init_state};

            svdat01.cuda_sv.applyOperation("CZ", {0, 1});
            svdat10.cuda_sv.applyOperation("CZ", {1, 0});

            svdat01.cuda_sv.CopyGpuDataToHost(svdat01.sv);
            svdat10.cuda_sv.CopyGpuDataToHost(svdat10.sv);

            CHECK(svdat01.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat10.sv.getDataVector() == Pennylane::approx(expected));
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyToffoli",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                             {false, false});
    const auto init_state = svdat.sv.getDataVector();

    SECTION("Apply directly") {
        SECTION("Toffoli 0,1,2 |+10> -> |010> + |111>") {
            std::vector<cp_t> expected{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::INVSQRT2<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::INVSQRT2<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat012{num_qubits, init_state};

            svdat012.cuda_sv.applyToffoli({0, 1, 2}, false);

            svdat012.cuda_sv.CopyGpuDataToHost(svdat012.sv);

            CHECK(svdat012.sv.getDataVector() == Pennylane::approx(expected));
        }

        SECTION("Toffoli 1,0,2 |+10> -> |010> + |111>") {
            std::vector<cp_t> expected{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0)};

            SVDataGPU<TestType> svdat102{num_qubits, init_state};

            svdat102.cuda_sv.applyToffoli({1, 0, 2}, false);
            svdat102.cuda_sv.CopyGpuDataToHost(svdat102.sv);
            CHECK(svdat102.sv.getDataVector() == Pennylane::approx(expected));
        }
        SECTION("Toffoli 0,2,1 |+10> -> |+10>") {
            auto &&expected = init_state;

            SVDataGPU<TestType> svdat021{num_qubits, init_state};

            svdat021.cuda_sv.applyToffoli({0, 2, 1}, false);

            svdat021.cuda_sv.CopyGpuDataToHost(svdat021.sv);
            CHECK(svdat021.sv.getDataVector() == Pennylane::approx(expected));
        }
        SECTION("Toffoli 1,2,0 |+10> -> |+10>") {
            auto &&expected = init_state;

            SVDataGPU<TestType> svdat120{num_qubits, init_state};

            svdat120.cuda_sv.applyToffoli({1, 2, 0}, false);
            svdat120.cuda_sv.CopyGpuDataToHost(svdat120.sv);
            CHECK(svdat120.sv.getDataVector() == Pennylane::approx(expected));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("Toffoli [0,1,2], [1,0,2] |+10> -> |+1+>") {
            std::vector<cp_t> expected{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0)};

            SVDataGPU<TestType> svdat012{num_qubits, init_state};
            SVDataGPU<TestType> svdat102{num_qubits, init_state};

            svdat012.cuda_sv.applyOperation("Toffoli", {0, 1, 2});
            svdat102.cuda_sv.applyOperation("Toffoli", {1, 0, 2});

            svdat012.cuda_sv.CopyGpuDataToHost(svdat012.sv);
            svdat102.cuda_sv.CopyGpuDataToHost(svdat102.sv);

            CHECK(svdat012.sv.getDataVector() == Pennylane::approx(expected));
            CHECK(svdat102.sv.getDataVector() == Pennylane::approx(expected));
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyCSWAP",
                   "[StateVectorCudaManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                 {false, false});
    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
    const auto init_state = svdat.sv.getDataVector();

    SECTION("Apply directly") {
        SECTION("CSWAP 0,1,2 |+10> -> |010> + |101>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};
            SVDataGPU<TestType> svdat012{num_qubits, init_state};

            svdat012.cuda_sv.applyCSWAP({0, 1, 2}, false);
            svdat012.cuda_sv.CopyGpuDataToHost(svdat012.sv);
            CHECK(svdat012.sv.getDataVector() == Pennylane::approx(expected));
        }

        SECTION("CSWAP 1,0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            SVDataGPU<TestType> svdat102{num_qubits, init_state};

            svdat102.cuda_sv.applyCSWAP({1, 0, 2}, false);
            svdat102.cuda_sv.CopyGpuDataToHost(svdat102.sv);
            CHECK(svdat102.sv.getDataVector() == Pennylane::approx(expected));
        }
        SECTION("CSWAP 2,1,0 |+10> -> |+10>") {
            auto &&expected = init_state;

            SVDataGPU<TestType> svdat021{num_qubits, init_state};

            svdat021.cuda_sv.applyCSWAP({2, 1, 0}, false);
            svdat021.cuda_sv.CopyGpuDataToHost(svdat021.sv);
            CHECK(svdat021.sv.getDataVector() == Pennylane::approx(expected));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CSWAP 0,1,2 |+10> -> |010> + |101>") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};
            SVDataGPU<TestType> svdat012{num_qubits, init_state};

            svdat012.cuda_sv.applyOperation("CSWAP", {0, 1, 2});
            svdat012.cuda_sv.CopyGpuDataToHost(svdat012.sv);
            CHECK(svdat012.sv.getDataVector() == Pennylane::approx(expected));
        }
    }
}
