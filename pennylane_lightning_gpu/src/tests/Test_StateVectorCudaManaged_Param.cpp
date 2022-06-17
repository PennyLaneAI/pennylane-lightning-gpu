
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

TEMPLATE_TEST_CASE("LightningGPU::applyRX", "[LightningGPU_Param]", double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 1;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{{0.1}, {0.6}};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.9987502603949663, 0.0},
                          {0.0, -0.04997916927067834}},
        std::vector<cp_t>{{0.9553364891256061, 0.0}, {0, -0.2955202066613395}},
        std::vector<cp_t>{{0.49757104789172696, 0.0}, {0, -0.867423225594017}}};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>{{0.9987502603949663, 0.0},
                          {0.0, 0.04997916927067834}},
        std::vector<cp_t>{{0.9553364891256061, 0.0}, {0, 0.2955202066613395}},
        std::vector<cp_t>{{0.49757104789172696, 0.0}, {0, 0.867423225594017}}};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("adj = false") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};

                svdat_direct.cuda_sv.applyRX({0}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.cuda_sv.applyOperation("RX", {0}, false,
                                                      {angles[index]});
                svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
                CHECK(svdat_dispatch.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
    }
    SECTION("adj = true") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyRX({0}, true, {angles[index]});
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.cuda_sv.applyOperation("RX", {0}, true,
                                                      {angles[index]});
                svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
                CHECK(svdat_dispatch.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyRY", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 1;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.8731983044562817, 0.04786268954660339},
                          {0.0876120655431924, -0.47703040785184303}},
        std::vector<cp_t>{{0.8243771119105122, 0.16439396602553008},
                          {0.3009211363333468, -0.45035926880694604}},
        std::vector<cp_t>{{0.10575112905629831, 0.47593196040758534},
                          {0.8711876098966215, -0.0577721051072477}}};
    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>{{0.8731983044562817, -0.04786268954660339},
                          {-0.0876120655431924, -0.47703040785184303}},
        std::vector<cp_t>{{0.8243771119105122, -0.16439396602553008},
                          {-0.3009211363333468, -0.45035926880694604}},
        std::vector<cp_t>{{0.10575112905629831, -0.47593196040758534},
                          {-0.8711876098966215, -0.0577721051072477}}};

    const std::vector<cp_t> init_state{{0.8775825618903728, 0.0},
                                       {0.0, -0.47942553860420306}};
    SECTION("adj = false") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};

                svdat_direct.cuda_sv.applyRY({0}, false, {angles[index]});
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.cuda_sv.applyOperation("RY", {0}, false,
                                                      {angles[index]});
                svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
                CHECK(svdat_dispatch.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
    }
    SECTION("adj = true") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};

                svdat_direct.cuda_sv.applyRY({0}, true, {angles[index]});
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);

                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.cuda_sv.applyOperation("RY", {0}, true,
                                                      {angles[index]});

                svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
                CHECK(svdat_dispatch.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyRZ", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});
    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> rz_data;
    rz_data.reserve(angles.size());
    for (auto &a : angles) {
        rz_data.push_back(Gates::getRZ<TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {rz_data[0][0], rz_data[0][0], rz_data[0][0], rz_data[0][0],
         rz_data[0][3], rz_data[0][3], rz_data[0][3], rz_data[0][3]},
        {
            rz_data[1][0],
            rz_data[1][0],
            rz_data[1][3],
            rz_data[1][3],
            rz_data[1][0],
            rz_data[1][0],
            rz_data[1][3],
            rz_data[1][3],
        },
        {rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3],
         rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_direct{num_qubits, init_state};

            svdat_direct.cuda_sv.applyRZ({index}, false, {angles[index]});
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
            svdat_dispatch.cuda_sv.applyOperation("RZ", {index}, false,
                                                  {angles[index]});

            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyPhaseShift", "[LightningGPU_Param]",
                   float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});
    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][3], ps_data[0][3], ps_data[0][3], ps_data[0][3]},
        {
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
        },
        {ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3],
         ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_direct{num_qubits, init_state};

            svdat_direct.cuda_sv.applyPhaseShift({index}, false,
                                                 {angles[index]});
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
            svdat_dispatch.cuda_sv.applyOperation("PhaseShift", {index}, false,
                                                  {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyControlledPhaseShift",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.cuda_sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                 {{0}, {1}, {2}}, {{false}, {false}, {false}});

    svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
    const std::vector<TestType> angles{0.3, 2.4};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3],
         ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        SVDataGPU<TestType> svdat_direct{num_qubits, init_state};

        svdat_direct.cuda_sv.applyControlledPhaseShift({0, 1}, false,
                                                       {angles[0]});
        svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
        CHECK(svdat_direct.sv.getDataVector() ==
              Pennylane::approx(expected_results[0]));
    }
    SECTION("Apply using dispatcher") {
        SVDataGPU<TestType> svdat_dispatch{num_qubits, init_state};
        svdat_dispatch.cuda_sv.applyOperation("ControlledPhaseShift", {1, 2},
                                              false, {angles[1]});
        svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
        CHECK(svdat_dispatch.sv.getDataVector() ==
              Pennylane::approx(expected_results[1]));
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyRot", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<std::vector<TestType>> angles{
        std::vector<TestType>{0.3, 0.8, 2.4},
        std::vector<TestType>{0.5, 1.1, 3.0},
        std::vector<TestType>{2.3, 0.1, 0.4}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat =
            Gates::getRot<TestType>(angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_direct{num_qubits};

            svdat_direct.cuda_sv.applyRot({index}, false, angles[index][0],
                                          angles[index][1], angles[index][2]);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation("Rot", {index}, false,
                                                  angles[index]);
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyCRot", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results(8);
    const auto rot_mat =
        Gates::getRot<TestType>(angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    const auto init_state = svdat.sv.getDataVector();

    SECTION("Apply directly") {
        SECTION("CRot0,1 |000> -> |000>") {
            SVDataGPU<TestType> svdat_direct{num_qubits};
            svdat_direct.cuda_sv.applyCRot({0, 1}, false, angles[0], angles[1],
                                           angles[2]);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(init_state));
        }
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            SVDataGPU<TestType> svdat_direct{num_qubits};
            svdat_direct.cuda_sv.applyOperation("PauliX", {0});

            svdat_direct.cuda_sv.applyCRot({0, 1}, false, angles[0], angles[1],
                                           angles[2]);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            SVDataGPU<TestType> svdat_direct{num_qubits};
            svdat_direct.cuda_sv.applyOperation("PauliX", {0});

            svdat_direct.cuda_sv.applyOperation("CRot", {0, 1}, false, angles);
            svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
            CHECK(svdat_direct.sv.getDataVector() ==
                  Pennylane::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingXX", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, 0.0};
    expected_results[0][6] = {0.0, -0.14943813247359922};

    expected_results[1][0] = {0.9210609940028851, 0.0};
    expected_results[1][6] = {0.0, -0.3894183423086505};

    expected_results[2][0] = {0.9887710779360422, 0.0};
    expected_results[2][5] = {0.0, -0.14943813247359922};

    expected_results[3][0] = {0.9210609940028851, 0.0};
    expected_results[3][5] = {0.0, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.0};
    expected_results_adj[0][6] = {0.0, 0.14943813247359922};

    expected_results_adj[1][0] = {0.9210609940028851, 0.0};
    expected_results_adj[1][6] = {0.0, 0.3894183423086505};

    expected_results_adj[2][0] = {0.9887710779360422, 0.0};
    expected_results_adj[2][5] = {0.0, 0.14943813247359922};

    expected_results_adj[3][0] = {0.9210609940028851, 0.0};
    expected_results_adj[3][5] = {0.0, 0.3894183423086505};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("IsingXX 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingXX({0, 1}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
        SECTION("IsingXX 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingXX({0, 2}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(
                    svdat_direct.sv.getDataVector() ==
                    Pennylane::approx(expected_results[index + angles.size()]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingXX 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingXX({0, 1}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingXX 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingXX({0, 2}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(
                          expected_results_adj[index + angles.size()]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation("IsingXX", {0, 1}, true,
                                                  {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingYY", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, 0.0};
    expected_results[0][6] = {0.0, 0.14943813247359922};

    expected_results[1][0] = {0.9210609940028851, 0.0};
    expected_results[1][6] = {0.0, 0.3894183423086505};

    expected_results[2][0] = {0.9887710779360422, 0.0};
    expected_results[2][5] = {0.0, 0.14943813247359922};

    expected_results[3][0] = {0.9210609940028851, 0.0};
    expected_results[3][5] = {0.0, 0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.0};
    expected_results_adj[0][6] = {0.0, -0.14943813247359922};

    expected_results_adj[1][0] = {0.9210609940028851, 0.0};
    expected_results_adj[1][6] = {0.0, -0.3894183423086505};

    expected_results_adj[2][0] = {0.9887710779360422, 0.0};
    expected_results_adj[2][5] = {0.0, -0.14943813247359922};

    expected_results_adj[3][0] = {0.9210609940028851, 0.0};
    expected_results_adj[3][5] = {0.0, -0.3894183423086505};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("IsingYY 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingYY({0, 1}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
        SECTION("IsingYY 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingYY({0, 2}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(
                    svdat_direct.sv.getDataVector() ==
                    Pennylane::approx(expected_results[index + angles.size()]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingYY 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingYY({0, 1}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingYY 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingYY({0, 2}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(
                          expected_results_adj[index + angles.size()]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation("IsingYY", {0, 1}, true,
                                                  {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingZZ", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0, 0}),
        std::vector<cp_t>(1 << num_qubits, {0, 0})};
    expected_results[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, 0.3894183423086505};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("IsingZZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingZZ({0, 1}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
        SECTION("IsingZZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingZZ({0, 2}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingZZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingZZ({0, 1}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingZZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyIsingZZ({0, 2}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation("IsingZZ", {0, 1}, true,
                                                  {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applySingleExcitation",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<cp_t> expected_results(1 << num_qubits);
    expected_results[0] = {1.0, 0.0};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        SECTION("SingleExcitation 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitation({0, 1}, false,
                                                           angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results));
            }
        }
        SECTION("SingleExcitation 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitation({0, 2}, false,
                                                           angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation("SingleExcitation", {0, 1},
                                                  false, {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applySingleExcitationMinus",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, 0.3894183423086505};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("SingleExcitationMinus 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitationMinus({0, 1}, false,
                                                                angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
        SECTION("SingleExcitationMinus 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitationMinus({0, 2}, false,
                                                                angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("SingleExcitationMinus 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitationMinus({0, 1}, true,
                                                                angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
        SECTION("SingleExcitationMinus 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitationMinus({0, 2}, true,
                                                                angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation(
                "SingleExcitationMinus", {0, 1}, true, {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applySingleExcitationPlus",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, 0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, -0.3894183423086505};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("SingleExcitationPlus 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitationPlus({0, 1}, false,
                                                               angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
        SECTION("SingleExcitationPlus 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitationPlus({0, 2}, false,
                                                               angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("SingleExcitationPlus 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitationPlus({0, 1}, true,
                                                               angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
        SECTION("SingleExcitationPlus 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applySingleExcitationPlus({0, 2}, true,
                                                               angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation(
                "SingleExcitationPlus", {0, 1}, true, {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyDoubleExcitation",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 4;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results(1 << num_qubits);
    expected_results[0] = {1.0, 0.0};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly") {
        SECTION("DoubleExcitation 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyDoubleExcitation({0, 1, 2, 3}, false,
                                                           angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation(
                "DoubleExcitation", {0, 1, 2, 3}, false, {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyDoubleExcitationMinus",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 4;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, 0.3894183423086505};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("DoubleExcitationMinus 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyDoubleExcitationMinus(
                    {0, 1, 2, 3}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("DoubleExcitationMinus 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyDoubleExcitationMinus(
                    {0, 1, 2, 3}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation(
                "DoubleExcitationMinus", {0, 1, 2, 3}, true, {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyDoubleExcitationPlus",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 4;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, 0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, -0.3894183423086505};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("DoubleExcitationPlus 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyDoubleExcitationPlus(
                    {0, 1, 2, 3}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("DoubleExcitationPlus 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyDoubleExcitationPlus(
                    {0, 1, 2, 3}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation(
                "DoubleExcitationPlus", {0, 1, 2, 3}, true, {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyMultiRZ", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVDataGPU<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, 0.3894183423086505};

    const auto init_state = svdat.sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("MultiRZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyMultiRZ({0, 1}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
        SECTION("MultiRZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyMultiRZ({0, 2}, false, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("MultiRZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyMultiRZ({0, 1}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
        SECTION("MultiRZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVDataGPU<TestType> svdat_direct{num_qubits, init_state};
                svdat_direct.cuda_sv.applyMultiRZ({0, 2}, true, angles[index]);
                svdat_direct.cuda_sv.CopyGpuDataToHost(svdat_direct.sv);
                CHECK(svdat_direct.sv.getDataVector() ==
                      Pennylane::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            SVDataGPU<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.cuda_sv.applyOperation("MultiRZ", {0, 1}, true,
                                                  {angles[index]});
            svdat_dispatch.cuda_sv.CopyGpuDataToHost(svdat_dispatch.sv);
            CHECK(svdat_dispatch.sv.getDataVector() ==
                  Pennylane::approx(expected_results_adj[index]));
        }
    }
}

// NOLINTNEXTLINE: Avoid complexity errors
TEMPLATE_TEST_CASE("LightningGPU::applyOperation 1 wire",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 5;

    // Note: gates are defined as right-to-left order

    SECTION("Apply XZ gate") {
        const std::vector<cp_t> xz_gate{
            cuUtil::ZERO<cp_t>(), cuUtil::ONE<cp_t>(), -cuUtil::ONE<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            SVDataGPU<TestType> svdat{num_qubits};
            SVDataGPU<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.cuda_sv.applyOperation({{"PauliX"}, {"PauliZ"}},
                                                      {{index}, {index}},
                                                      {false, false});

                svdat.cuda_sv.applyOperation_std("XZ", {index}, false, {0.0},
                                                 xz_gate);
            }
            svdat_expected.cuda_sv.CopyGpuDataToHost(svdat_expected.sv);
            svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);

            CHECK(svdat.sv.getDataVector() ==
                  svdat_expected.sv.getDataVector());
        }
    }
    SECTION("Apply ZX gate") {
        const std::vector<cp_t> zx_gate{
            cuUtil::ZERO<cp_t>(), -cuUtil::ONE<cp_t>(), cuUtil::ONE<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            SVDataGPU<TestType> svdat{num_qubits};
            SVDataGPU<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.cuda_sv.applyOperation({{"PauliZ"}, {"PauliX"}},
                                                      {{index}, {index}},
                                                      {false, false});
                svdat_expected.cuda_sv.CopyGpuDataToHost(svdat_expected.sv);
                svdat.cuda_sv.applyOperation_std("ZX", {index}, false, {0.0},
                                                 zx_gate);
                svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
            }
            CHECK(svdat.sv.getDataVector() ==
                  svdat_expected.sv.getDataVector());
        }
    }
    SECTION("Apply XY gate") {
        const std::vector<cp_t> xy_gate{
            -cuUtil::IMAG<cp_t>(), cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::IMAG<cp_t>()};

        SECTION("Apply using dispatcher") {
            SVDataGPU<TestType> svdat{num_qubits};
            SVDataGPU<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.cuda_sv.applyOperation({{"PauliX"}, {"PauliY"}},
                                                      {{index}, {index}},
                                                      {false, false});
                svdat_expected.cuda_sv.CopyGpuDataToHost(svdat_expected.sv);
                svdat.cuda_sv.applyOperation_std("XY", {index}, false, {0.0},
                                                 xy_gate);
                svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
            }
            CHECK(svdat.sv.getDataVector() ==
                  svdat_expected.sv.getDataVector());
        }
    }
    SECTION("Apply YX gate") {
        const std::vector<cp_t> yx_gate{
            cuUtil::IMAG<cp_t>(), cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            -cuUtil::IMAG<cp_t>()};

        SECTION("Apply using dispatcher") {
            SVDataGPU<TestType> svdat{num_qubits};
            SVDataGPU<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.cuda_sv.applyOperation({{"PauliY"}, {"PauliX"}},
                                                      {{index}, {index}},
                                                      {false, false});
                svdat_expected.cuda_sv.CopyGpuDataToHost(svdat_expected.sv);
                svdat.cuda_sv.applyOperation_std("YX", {index}, false, {0.0},
                                                 yx_gate);
                svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
            }
            CHECK(svdat.sv.getDataVector() ==
                  svdat_expected.sv.getDataVector());
        }
    }
    SECTION("Apply YZ gate") {
        const std::vector<cp_t> yz_gate{
            cuUtil::ZERO<cp_t>(), -cuUtil::IMAG<cp_t>(), -cuUtil::IMAG<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            SVDataGPU<TestType> svdat{num_qubits};
            SVDataGPU<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.cuda_sv.applyOperation({{"PauliY"}, {"PauliZ"}},
                                                      {{index}, {index}},
                                                      {false, false});
                svdat_expected.cuda_sv.CopyGpuDataToHost(svdat_expected.sv);
                svdat.cuda_sv.applyOperation_std("YZ", {index}, false, {0.0},
                                                 yz_gate);
                svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
            }
            CHECK(svdat.sv.getDataVector() ==
                  svdat_expected.sv.getDataVector());
        }
    }
    SECTION("Apply ZY gate") {
        const std::vector<cp_t> zy_gate{
            cuUtil::ZERO<cp_t>(), cuUtil::IMAG<cp_t>(), cuUtil::IMAG<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            SVDataGPU<TestType> svdat{num_qubits};
            SVDataGPU<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.cuda_sv.applyOperation({{"PauliZ"}, {"PauliY"}},
                                                      {{index}, {index}},
                                                      {false, false});
                svdat_expected.cuda_sv.CopyGpuDataToHost(svdat_expected.sv);
                svdat.cuda_sv.applyOperation_std("ZY", {index}, false, {0.0},
                                                 zy_gate);
                svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
            }
            CHECK(svdat.sv.getDataVector() ==
                  svdat_expected.sv.getDataVector());
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyOperation multiple wires",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;

    SVDataGPU<TestType> svdat_init{num_qubits};
    svdat_init.cuda_sv.applyOperation(
        {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
        {false, false, false});

    svdat_init.cuda_sv.CopyGpuDataToHost(svdat_init.sv);
    const auto cz_gate = cuGates::getCZ<cp_t>();
    const auto tof_gate = cuGates::getToffoli<cp_t>();
    const auto arb_gate = cuGates::getToffoli<cp_t>();

    SECTION("Apply CZ gate") {
        SVDataGPU<TestType> svdat{num_qubits, svdat_init.sv.getDataVector()};
        SVDataGPU<TestType> svdat_expected{num_qubits,
                                           svdat_init.sv.getDataVector()};

        svdat_expected.cuda_sv.applyOperation(
            {{"Hadamard"}, {"CNOT"}, {"Hadamard"}}, {{1}, {0, 1}, {1}},
            {false, false, false});

        svdat_expected.cuda_sv.CopyGpuDataToHost(svdat_expected.sv);
        svdat.cuda_sv.applyOperation_std("CZmat", {0, 1}, false, {0.0},
                                         cz_gate);
        svdat.cuda_sv.CopyGpuDataToHost(svdat.sv);
        CHECK(svdat.sv.getDataVector() ==
              Pennylane::approx(svdat_expected.sv.getDataVector()));
    }
}

TEMPLATE_TEST_CASE("Sample", "[LightningGPU_Param]", float, double) {
    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    // Defining the State Vector that will be measured.
    size_t num_qubits = 3;
    size_t data_size = std::pow(2, num_qubits);

    std::vector<std::complex<TestType>> init_state(data_size, 0);
    init_state[0] = 1;
    SVDataGPU<TestType> svdat{num_qubits, init_state};
    TestType alpha = 0.7;
    TestType beta = 0.5;
    TestType gamma = 0.2;
    svdat.sv.applyOperations(
        {"RX", "RY", "RX", "RY", "RX", "RY"}, {{0}, {0}, {1}, {1}, {2}, {2}},
        {false, false, false, false, false, false},
        {{alpha}, {alpha}, {beta}, {beta}, {gamma}, {gamma}});
    svdat.cuda_sv.CopyHostDataToGpu(svdat.sv);
    std::vector<TestType> expected_probabilities = {
        0.687573, 0.013842, 0.089279, 0.001797,
        0.180036, 0.003624, 0.023377, 0.000471};

    size_t N = std::pow(2, num_qubits);
    size_t num_samples = 100000;
    auto &&samples = svdat.cuda_sv.generate_samples(num_samples);

    std::vector<size_t> counts(N, 0);
    std::vector<size_t> samples_decimal(num_samples, 0);

    // convert samples to decimal and then bin them in counts
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < num_qubits; j++) {
            if (samples[i * num_qubits + j] != 0) {
                samples_decimal[i] += twos[(num_qubits - 1 - j)];
            }
        }
        counts[samples_decimal[i]] += 1;
    }

    // compute estimated probabilities from histogram
    std::vector<TestType> probabilities(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (TestType)num_samples;
    }

    SECTION("No wires provided:") {
        REQUIRE_THAT(probabilities,
                     Catch::Approx(expected_probabilities).margin(.05));
    }
}
