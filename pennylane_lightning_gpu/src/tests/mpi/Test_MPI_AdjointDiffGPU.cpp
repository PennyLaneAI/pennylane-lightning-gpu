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

#include "../TestHelpers.hpp"
#include "AdjointDiffGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "Util.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

using namespace Pennylane::CUDA;
using namespace Pennylane::Algorithms;

TEST_CASE(
    "AdjointJacobianGPUMPI::AdjointJacobianGPUMPI Op=[RX,RX,RX], Obs=[Z,Z,Z]",
    "[AdjointJacobianGPUMPI]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};

    const size_t num_qubits = 3;
    const size_t num_obs = 3;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    std::vector<std::vector<double>> jacobian(
        num_obs, std::vector<double>(tp.size(), 0));

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    AdjointJacobianGPUMPI<double, StateVectorCudaMPI> adj;
    {
        StateVectorCudaMPI<double> sv_ref(mpi_manager, dt_local, 4,
                                          nGlobalIndexBits, nLocalIndexBits);
        sv_ref.initSV_MPI();

        const auto obs1 = std::make_shared<NamedObsGPUMPI<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsGPUMPI<double>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObsGPUMPI<double>>(
            "PauliZ", std::vector<size_t>{2});
        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(sv_ref, jacobian, {obs1, obs2, obs3}, ops, tp,
                            true);

        CAPTURE(jacobian);

        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(-sin(param[1]) == Approx(jacobian[1][1]).margin(1e-7));
        CHECK(-sin(param[2]) == Approx(jacobian[2][2]).margin(1e-7));
    }
}

TEST_CASE(
    "AdjointJacobianGPUMPI::AdjointJacobianGPUMPI Op=[RX,RX,RX], Obs=[Z,Z,Z],"
    "TParams=[0,2]",
    "[AdjointJacobianGPUMPI]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 2};

    const size_t num_qubits = 3;
    const size_t num_obs = 3;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    std::vector<std::vector<double>> jacobian(
        num_obs, std::vector<double>(tp.size(), 0));

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    AdjointJacobianGPUMPI<double, StateVectorCudaMPI> adj;
    {
        StateVectorCudaMPI<double> sv_ref(mpi_manager, dt_local, 4,
                                          nGlobalIndexBits, nLocalIndexBits);
        sv_ref.initSV_MPI();

        const auto obs1 = std::make_shared<NamedObsGPUMPI<double>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsGPUMPI<double>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObsGPUMPI<double>>(
            "PauliZ", std::vector<size_t>{2});
        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(sv_ref, jacobian, {obs1, obs2, obs3}, ops, tp,
                            true);

        CAPTURE(jacobian);

        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(0 == Approx(jacobian[1][1]).margin(1e-7));
        CHECK(-sin(param[2]) == Approx(jacobian[2][1]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPUMPI::adjointJacobianMPI Op=Mixed, Obs=[XXX]",
          "[AdjointJacobianGPUMPI]") {
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2, 3, 4, 5};

    const size_t num_qubits = 3;
    const size_t num_obs = 1;

    std::vector<std::vector<double>> jacobian(
        num_obs, std::vector<double>(tp.size(), 0));

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    AdjointJacobianGPUMPI<double, StateVectorCudaMPI> adj;
    {
        StateVectorCudaMPI<double> sv_ref(mpi_manager, dt_local, 4,
                                          nGlobalIndexBits, nLocalIndexBits);
        sv_ref.initSV_MPI();

        const auto obs = std::make_shared<TensorProdObsGPUMPI<double>>(
            std::make_shared<NamedObsGPUMPI<double>>("PauliX",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObsGPUMPI<double>>("PauliX",
                                                     std::vector<size_t>{1}),
            std::make_shared<NamedObsGPUMPI<double>>("PauliX",
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

        adj.adjointJacobian(sv_ref, jacobian, {obs}, ops, tp, true);

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