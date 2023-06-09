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

#include "AdjointDiffGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "../TestHelpers.hpp"
#include "Util.hpp"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

using namespace Pennylane::CUDA;
using namespace Pennylane::Algorithms;


TEST_CASE("AdjointJacobianGPUMPI::AdjointJacobianGPUMPI Op=[RX,RX,RX], Obs=[Z,Z,Z]",
          "[AdjointJacobianGPUMPI]") {

    AdjointJacobianGPUMPI<double,StateVectorCudaMPI> adj;
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
    {
        StateVectorCudaMPI<double> sv_ref(mpi_manager, dt_local, 0, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv_ref.initSV_MPI();

        const auto obs1 = std::make_shared<NamedObsGPUMPI<double, StateVectorCudaMPI>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsGPUMPI<double, StateVectorCudaMPI>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObsGPUMPI<double, StateVectorCudaMPI>>(
            "PauliZ", std::vector<size_t>{2});
        auto ops = adj.createOpsData({"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false});

        adj.adjointJacobian(sv_ref, jacobian, {obs1, obs2, obs3}, ops, tp, true, dt_local);

        //CAPTURE(jacobian);
        mpi_manager.Barrier();

        // Computed with parameter shift
        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(-sin(param[1]) == Approx(jacobian[1][1]).margin(1e-7));
        CHECK(-sin(param[2]) == Approx(jacobian[2][2]).margin(1e-7));
    }
}
