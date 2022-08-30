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

/**
 * @brief Tests the constructability of the AdjointDiff.hpp classes.
 *
 */
TEMPLATE_TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU",
                   "[AdjointJacobianGPU]", float, double) {
    SECTION("AdjointJacobianGPU") {
        REQUIRE(std::is_constructible<AdjointJacobianGPU<>>::value);
    }
    SECTION("AdjointJacobianGPU<TestType> {}") {
        REQUIRE(std::is_constructible<AdjointJacobianGPU<TestType>>::value);
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Op=RX, Obs=Z",
          "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 1;
        const size_t num_obs = 1;
        const auto obs = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{0});
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        for (const auto &p : param) {
            auto ops = adj.createOpsData({"RX"}, {{p}}, {{0}}, {false});

            SVDataGPU<double> psi(num_qubits);
            adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                                jacobian, {obs}, ops, tp, true);
            CAPTURE(jacobian);
            CHECK(-sin(p) == Approx(jacobian[0].front()));
        }
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Op=RY, Obs=X",
          "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 1;
        const size_t num_obs = 1;

        const auto obs = std::make_shared<NamedObsGPU<double>>(
            "PauliX", std::vector<size_t>{0});
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        for (const auto &p : param) {
            auto ops = adj.createOpsData({"RY"}, {{p}}, {{0}}, {false});

            SVDataGPU<double> psi(num_qubits);
            adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                                jacobian, {obs}, ops, tp, true);

            CAPTURE(jacobian);
            CHECK(cos(p) == Approx(jacobian[0].front()).margin(1e-7));
        }
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Op=RX, Obs=[Z,Z]",
          "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 2;
        const size_t num_obs = 2;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        SVDataGPU<double> psi(num_qubits);

        const auto obs1 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{1});

        auto ops = adj.createOpsData({"RX"}, {{param[0]}}, {{0}}, {false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian, {obs1, obs2}, ops, tp, true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(0.0 == Approx(jacobian[1][0]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Op=[RX,RX,RX], Obs=[Z,Z,Z]",
          "[AdjointJacobianGPU]") {

    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 3;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        SVDataGPU<double> psi(num_qubits);

        const auto obs1 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{2});
        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian, {obs1, obs2, obs3}, ops, tp, true);

        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(-sin(param[1]) == Approx(jacobian[1][1]).margin(1e-7));
        CHECK(-sin(param[2]) == Approx(jacobian[2][2]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Op=[RX,RX,RX], Obs=[Z,Z,Z],"
          "TParams=[0,2]",
          "[AdjointJacobianGPU]") {

    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 3;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        SVDataGPU<double> psi(num_qubits);

        const auto obs1 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{2});
        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian, {obs1, obs2, obs3}, ops, tp, true);

        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(0 == Approx(jacobian[1][1]).margin(1e-7));
        CHECK(-sin(param[2]) == Approx(jacobian[2][1]).margin(1e-7));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=[RX,RX,RX], Obs=[ZZZ]",
          "[Algorithms]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        SVDataGPU<double> psi(num_qubits);

        const auto obs = std::make_shared<TensorProdObsGPU<double>>(
            std::make_shared<NamedObsGPU<double>>("PauliZ",
                                                  std::vector<size_t>{0}),
            std::make_shared<NamedObsGPU<double>>("PauliZ",
                                                  std::vector<size_t>{1}),
            std::make_shared<NamedObsGPU<double>>("PauliZ",
                                                  std::vector<size_t>{2}));
        auto ops = OpsData<double>({"RX", "RX", "RX"},
                                   {{param[0]}, {param[1]}, {param[2]}},
                                   {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian, {obs}, ops, tp, true);

        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-0.1755096592645253 == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(0.26478810666384334 == Approx(jacobian[0][1]).margin(1e-7));
        CHECK(-0.6312451595102775 == Approx(jacobian[0][2]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Op=Mixed, Obs=[XXX]",
          "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2, 3, 4, 5};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        SVDataGPU<double> psi(num_qubits);

        const auto obs = std::make_shared<TensorProdObsGPU<double>>(
            std::make_shared<NamedObsGPU<double>>("PauliX",
                                                  std::vector<size_t>{0}),
            std::make_shared<NamedObsGPU<double>>("PauliX",
                                                  std::vector<size_t>{1}),
            std::make_shared<NamedObsGPU<double>>("PauliX",
                                                  std::vector<size_t>{2}));
        auto ops = adj.createOpsData(
            {"RZ", "RY", "RZ", "CNOT", "CNOT", "RZ", "RY", "RZ"},
            {{param[0]},
             {param[1]},
             {param[2]},
             {},
             {},
             {param[0]},
             {param[1]},
             {param[2]}},
            {{0}, {0}, {0}, {0, 1}, {1, 2}, {1}, {1}, {1}},
            {false, false, false, false, false, false, false, false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian, {obs}, ops, tp, true);
        CAPTURE(jacobian);

        // Computed with PennyLane using default.qubit.adjoint_jacobian
        CHECK(0.0 == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(-0.674214427 == Approx(jacobian[0][1]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[0][2]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[0][3]).margin(1e-7));
        CHECK(-0.0129093062 == Approx(jacobian[0][4]).margin(1e-7));
        CHECK(0.323846156 == Approx(jacobian[0][5]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Decomposed Rot gate, non "
          "computational basis state",
          "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_qubits = 1;
        const size_t num_obs = 1;

        const auto thetas = Pennylane::Util::linspace(-2 * M_PI, 2 * M_PI, 7);
        std::unordered_map<double, std::vector<double>> expec_results{
            {thetas[0], {0, -9.90819496e-01, 0}},
            {thetas[1], {-8.18996553e-01, 1.62526544e-01, 0}},
            {thetas[2], {-0.203949, 0.48593716, 0}},
            {thetas[3], {0, 1, 0}},
            {thetas[4], {-2.03948985e-01, 4.85937177e-01, 0}},
            {thetas[5], {-8.18996598e-01, 1.62526487e-01, 0}},
            {thetas[6], {0, -9.90819511e-01, 0}}};

        for (const auto &theta : thetas) {
            std::vector<double> local_params{theta, std::pow(theta, 3),
                                             Pennylane::Util::SQRT2<double>() *
                                                 theta};
            std::vector<std::vector<double>> jacobian(
                num_obs, std::vector<double>(tp.size(), 0));

            std::vector<std::complex<double>> cdata{
                {Pennylane::Util::INVSQRT2<double>()},
                {-Pennylane::Util::INVSQRT2<double>()}};
            std::vector<std::complex<double>> new_data{cdata.begin(),
                                                       cdata.end()};
            SVDataGPU<double> psi(num_qubits, new_data);

            const auto obs = std::make_shared<NamedObsGPU<double>>(
                "PauliZ", std::vector<size_t>{0});

            auto ops = adj.createOpsData(
                {"RZ", "RY", "RZ"},
                {{local_params[0]}, {local_params[1]}, {local_params[2]}},
                {{0}, {0}, {0}}, {false, false, false});

            adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                                jacobian, {obs}, ops, tp, true);
            CAPTURE(theta);
            CAPTURE(jacobian);

            // Computed with PennyLane using default.qubit
            CHECK(expec_results[theta][0] ==
                  Approx(jacobian[0][0]).margin(1e-7));
            CHECK(expec_results[theta][1] ==
                  Approx(jacobian[0][1]).margin(1e-7));
            CHECK(expec_results[theta][2] ==
                  Approx(jacobian[0][2]).margin(1e-7));
        }
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Mixed Ops, Obs and TParams",
          "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const size_t num_qubits = 2;
        const std::vector<size_t> t_params{1, 2, 3};
        const size_t num_obs = 1;

        const auto thetas = Pennylane::Util::linspace(-2 * M_PI, 2 * M_PI, 8);

        std::vector<double> local_params{0.543, 0.54, 0.1,  0.5, 1.3,
                                         -2.3,  0.5,  -0.5, 0.5};
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(t_params.size(), 0));

        std::vector<std::complex<double>> cdata{
            {Pennylane::Util::ONE<double>()},
            {Pennylane::Util::ZERO<double>()},
            {Pennylane::Util::ZERO<double>()},
            {Pennylane::Util::ZERO<double>()}};
        std::vector<std::complex<double>> new_data{cdata.begin(), cdata.end()};
        SVDataGPU<double> psi(num_qubits, new_data);

        const auto obs = std::make_shared<TensorProdObsGPU<double>>(
            std::make_shared<NamedObsGPU<double>>("PauliX",
                                                  std::vector<size_t>{0}),
            std::make_shared<NamedObsGPU<double>>("PauliZ",
                                                  std::vector<size_t>{1}));
        auto ops =
            adj.createOpsData({"Hadamard", "RX", "CNOT", "RZ", "RY", "RZ", "RZ",
                               "RY", "RZ", "RZ", "RY", "CNOT"},
                              {{},
                               {local_params[0]},
                               {},
                               {local_params[1]},
                               {local_params[2]},
                               {local_params[3]},
                               {local_params[4]},
                               {local_params[5]},
                               {local_params[6]},
                               {local_params[7]},
                               {local_params[8]},
                               {}},
                              std::vector<std::vector<std::size_t>>{{0},
                                                                    {0},
                                                                    {0, 1},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {1},
                                                                    {0, 1}},
                              {false, false, false, false, false, false, false,
                               false, false, false, false, false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian, {obs}, ops, t_params, true);

        std::vector<double> expected{-0.71429188, 0.04998561, -0.71904837};
        // Computed with PennyLane using default.qubit
        CHECK(expected[0] == Approx(jacobian[0][0]));
        CHECK(expected[1] == Approx(jacobian[0][1]));
        CHECK(expected[2] == Approx(jacobian[0][2]));
    }
}

TEST_CASE("AdjointJacobianGPU::batchAdjointJacobian Mixed Ops, Obs and TParams",
          "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const size_t num_qubits = 2;
        const std::vector<size_t> t_params{1, 2, 3};
        const size_t num_obs = 1;

        const auto thetas = Pennylane::Util::linspace(-2 * M_PI, 2 * M_PI, 8);

        std::vector<double> local_params{0.543, 0.54, 0.1,  0.5, 1.3,
                                         -2.3,  0.5,  -0.5, 0.5};
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(t_params.size(), 0));

        std::vector<std::complex<double>> cdata{
            {Pennylane::Util::ONE<double>()},
            {Pennylane::Util::ZERO<double>()},
            {Pennylane::Util::ZERO<double>()},
            {Pennylane::Util::ZERO<double>()}};
        SVDataGPU<double> psi(num_qubits, std::vector<std::complex<double>>{
                                              cdata.begin(), cdata.end()});

        const auto obs = std::make_shared<TensorProdObsGPU<double>>(
            std::make_shared<NamedObsGPU<double>>("PauliX",
                                                  std::vector<size_t>{0}),
            std::make_shared<NamedObsGPU<double>>("PauliZ",
                                                  std::vector<size_t>{1}));
        auto ops =
            adj.createOpsData({"Hadamard", "RX", "CNOT", "RZ", "RY", "RZ", "RZ",
                               "RY", "RZ", "RZ", "RY", "CNOT"},
                              {{},
                               {local_params[0]},
                               {},
                               {local_params[1]},
                               {local_params[2]},
                               {local_params[3]},
                               {local_params[4]},
                               {local_params[5]},
                               {local_params[6]},
                               {local_params[7]},
                               {local_params[8]},
                               {}},
                              std::vector<std::vector<std::size_t>>{{0},
                                                                    {0},
                                                                    {0, 1},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {1},
                                                                    {0, 1}},
                              {false, false, false, false, false, false, false,
                               false, false, false, false, false});

        adj.batchAdjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                                 jacobian, {obs}, ops, t_params, true);

        std::vector<double> expected{-0.71429188, 0.04998561, -0.71904837};
        // Computed with PennyLane using default.qubit
        CHECK(expected[0] == Approx(jacobian[0][0]));
        CHECK(expected[1] == Approx(jacobian[0][1]));
        CHECK(expected[2] == Approx(jacobian[0][2]));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=RX, Obs=Ham[Z0+Z1]", "[Algorithms]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 2;
        const size_t num_obs = 1;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        SVDataGPU<double> psi(num_qubits);

        const auto obs1 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{1});

        auto ham = HamiltonianGPU<double>::create({0.3, 0.7}, {obs1, obs2});

        auto ops = OpsData<double>({"RX"}, {{param[0]}}, {{0}}, {false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian, {ham}, ops, tp, true);

        CAPTURE(jacobian);
        CHECK(-0.3 * sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
    }
}

TEST_CASE(
    "AdjointJacobianGPU::AdjointJacobianGPU Op=[RX,RX,RX], Obs=Ham[Z0+Z1+Z2], "
    "TParams=[0,2]",
    "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> t_params{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(t_params.size(), 0));

        SVDataGPU<double> psi(num_qubits);

        auto obs1 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{0});
        auto obs2 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{1});
        auto obs3 = std::make_shared<NamedObsGPU<double>>(
            "PauliZ", std::vector<size_t>{2});

        auto ham = HamiltonianGPU<double>::create({0.47, 0.32, 0.96},
                                                  {obs1, obs2, obs3});

        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian, {ham}, ops, t_params, true);
        CAPTURE(jacobian);

        CHECK((-0.47 * sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7)));
        CHECK((-0.96 * sin(param[2]) == Approx(jacobian[0][1]).margin(1e-7)));
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Test HermitianObs",
          "[AdjointJacobianGPU]") {
    AdjointJacobianGPU<double> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> t_params{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;

        std::vector<std::vector<double>> jacobian1(
            num_obs, std::vector<double>(t_params.size(), 0));
        std::vector<std::vector<double>> jacobian2(
            num_obs, std::vector<double>(t_params.size(), 0));

        SVDataGPU<double> psi(num_qubits);

        auto obs1 = std::make_shared<TensorProdObsGPU<double>>(
            std::make_shared<NamedObsGPU<double>>("PauliZ",
                                                  std::vector<size_t>{0}),
            std::make_shared<NamedObsGPU<double>>("PauliZ",
                                                  std::vector<size_t>{1}));
        auto obs2 = std::make_shared<HermitianObsGPU<double>>(
            std::vector<std::complex<double>>{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1,
                                              0, 0, 0, 0, 1},
            std::vector<size_t>{0, 1});

        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian1, {obs1}, ops, t_params, true);
        adj.adjointJacobian(psi.cuda_sv.getData(), psi.cuda_sv.getLength(),
                            jacobian2, {obs2}, ops, t_params, true);

        CHECK((jacobian1[0] == PLApprox(jacobian2[0]).margin(1e-7)));
    }
}
