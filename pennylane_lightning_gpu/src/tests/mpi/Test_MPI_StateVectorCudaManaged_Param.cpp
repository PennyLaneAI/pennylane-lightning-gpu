#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>
#include <mpi.h>

#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"
#include "StateVectorRawCPU.hpp"

#include "MPIManager.hpp"

#include "../TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::MPI;
using namespace CUDA;

#define num_qubits 8
#define lsb_1qbit                                                              \
    { 0 }
#define msb_1qbit                                                              \
    { num_qubits - 1 }

#define lsb_2qbit                                                              \
    { 0, 1 }
#define msb_2qbit                                                              \
    { num_qubits - 2, num_qubits - 1 }
#define mlsb_2qbit                                                             \
    { 0, num_qubits - 1 }

#define lsb_3qbit                                                              \
    { 0, 1, 2 }
#define msb_3qbit                                                              \
    { num_qubits - 3, num_qubits - 2, num_qubits - 1 }
#define mlsb_3qbit                                                             \
    { 0, num_qubits - 2, num_qubits - 1 }

#define lsb_4qbit                                                              \
    { 0, 1, 2, 3 }
#define msb_4qbit                                                              \
    { num_qubits - 4, num_qubits - 3, num_qubits - 2, num_qubits - 1 }
#define mlsb_4qbit                                                             \
    { 0, 1, num_qubits - 2, num_qubits - 1 }

#define angle_1param                                                           \
    { 0.4 }
#define angle_3param                                                           \
    { 0.4, 0.3, 0.2 }

#define PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, NUM_QUBITS, GATE_METHOD,       \
                                      GATE_NAME, WIRE, ANGLE)                  \
    {                                                                          \
        using cp_t = std::complex<TestType>;                                   \
        using PrecisionT = TestType;                                           \
        MPIManager mpi_manager(MPI_COMM_WORLD);                                \
        int nGlobalIndexBits =                                                 \
            std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - \
            1;                                                                 \
        int nLocalIndexBits = NUM_QUBITS - nGlobalIndexBits;                   \
        int subSvLength = 1 << nLocalIndexBits;                                \
        int svLength = 1 << NUM_QUBITS;                                        \
        mpi_manager.Barrier();                                                 \
        std::vector<cp_t> init_sv(svLength);                                   \
        std::vector<cp_t> expected_sv(svLength);                               \
        if (mpi_manager.getRank() == 0) {                                      \
            std::mt19937 re{1337};                                             \
            auto random_sv = createRandomState<PrecisionT>(re, NUM_QUBITS);    \
            init_sv = random_sv;                                               \
        }                                                                      \
        auto local_state = mpi_manager.scatter(init_sv, 0);                    \
        mpi_manager.Barrier();                                                 \
        int nDevices = 0;                                                      \
        cudaGetDeviceCount(&nDevices);                                         \
        int deviceId = mpi_manager.getRank() % nDevices;                       \
        cudaSetDevice(deviceId);                                               \
        mpi_manager.Barrier();                                                 \
        SECTION("Apply directly") {                                            \
            SECTION("Operation on target") {                                   \
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits, \
                                                nLocalIndexBits);              \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.GATE_METHOD(WIRE, false, ANGLE);                            \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                SVDataGPU<TestType> svdat{NUM_QUBITS, init_sv};                \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.cuda_sv.GATE_METHOD(WIRE, false, ANGLE);             \
                    svdat.cuda_sv.CopyGpuDataToHost(                           \
                        expected_sv.data(),                                    \
                        static_cast<std::size_t>(svLength));                   \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state == Pennylane::approx(expected_local_sv));    \
            }                                                                  \
        }                                                                      \
        SECTION("Apply using dispatcher") {                                    \
            SECTION("Operation on target") {                                   \
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits, \
                                                nLocalIndexBits);              \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.applyOperation(GATE_NAME, WIRE, false, ANGLE);              \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                SVDataGPU<TestType> svdat{NUM_QUBITS, init_sv};                \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.cuda_sv.applyOperation(GATE_NAME, WIRE, false,       \
                                                 ANGLE);                       \
                    svdat.cuda_sv.CopyGpuDataToHost(                           \
                        expected_sv.data(),                                    \
                        static_cast<std::size_t>(svLength));                   \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state == Pennylane::approx(expected_local_sv));    \
            }                                                                  \
        }                                                                      \
    }

TEMPLATE_TEST_CASE("StateVectorCudaMPI::GateOpsParam",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRX, "RX",
                                  lsb_1qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRX, "RX",
                                  msb_1qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRY, "RY",
                                  lsb_1qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRY, "RY",
                                  msb_1qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRZ, "RZ",
                                  lsb_1qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRZ, "RZ",
                                  msb_1qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyPhaseShift,
                                  "PhaseShift", lsb_1qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyPhaseShift,
                                  "PhaseShift", msb_1qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRot, "Rot",
                                  lsb_1qbit, angle_3param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRot, "Rot",
                                  msb_1qbit, angle_3param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXX, "IsingXX",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXX, "IsingXX",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXX, "IsingXX",
                                  msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingYY, "IsingYY",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingYY, "IsingYY",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingYY, "IsingYY",
                                  msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingZZ, "IsingZZ",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingZZ, "IsingZZ",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingZZ, "IsingZZ",
                                  msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyControlledPhaseShift, "ControlledPhaseShift",
        lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyControlledPhaseShift, "ControlledPhaseShift",
        mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyControlledPhaseShift, "ControlledPhaseShift",
        msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRX, "CRX",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRX, "CRX",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRX, "CRX",
                                  msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRY, "CRY",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRY, "CRY",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRY, "CRY",
                                  msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRZ, "CRZ",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRZ, "CRZ",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRZ, "CRZ",
                                  msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRot, "CRot",
                                  lsb_2qbit, angle_3param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRot, "CRot",
                                  mlsb_2qbit, angle_3param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRot, "CRot",
                                  msb_2qbit, angle_3param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applySingleExcitation,
                                  "SingleExcitation", lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applySingleExcitation,
                                  "SingleExcitation", mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applySingleExcitation,
                                  "SingleExcitation", msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationMinus,
        "SingleExcitationMinus", lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationMinus,
        "SingleExcitationMinus", mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationMinus,
        "SingleExcitationMinus", msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationPlus, "SingleExcitationPlus",
        lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationPlus, "SingleExcitationPlus",
        mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationPlus, "SingleExcitationPlus",
        msb_2qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyDoubleExcitation,
                                  "DoubleExcitation", lsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyDoubleExcitation,
                                  "DoubleExcitation", mlsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyDoubleExcitation,
                                  "DoubleExcitation", msb_4qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationMinus,
        "DoubleExcitationMinus", lsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationMinus,
        "DoubleExcitationMinus", mlsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationMinus,
        "DoubleExcitationMinus", msb_4qbit, angle_1param);

    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationPlus, "DoubleExcitationPlus",
        lsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationPlus, "DoubleExcitationPlus",
        mlsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationPlus, "DoubleExcitationPlus",
        msb_4qbit, angle_1param);
}
/*
TEMPLATE_TEST_CASE("StateVectorCudaMPI::GateOps_Param",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    using PrecisionT = TestType;
    const std::size_t num_qubits = 6;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    int svLength = 1 << num_qubits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv(svLength);
    std::vector<cp_t> expected_sv(svLength);

    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        auto random_sv = createRandomState<PrecisionT>(re, num_qubits);
        init_sv = random_sv;
    }

    auto local_state = mpi_manager.scatter(init_sv, 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{0.4, 0.3, 0.2};

    SECTION("Gate: RX") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyRX({0}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyRX({0}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyRX({num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyRX({num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("RX", {0}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("RX", {0}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("RX", {num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("RX", {num_qubits - 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: RY") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyRY({0}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyRY({0}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyRY({num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyRY({num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("RY", {0}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("RY", {0}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("RY", {num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("RY", {num_qubits - 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: RZ") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyRZ({0}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyRZ({0}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyRZ({num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyRZ({num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("RZ", {0}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("RZ", {0}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("RZ", {num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("RZ", {num_qubits - 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: PhaseShift") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyPhaseShift({0}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyPhaseShift({0}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyPhaseShift({num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyPhaseShift({num_qubits - 1}, false,
                                                  {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("PhaseShift", {0}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("PhaseShift", {0}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("PhaseShift", {num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("PhaseShift", {num_qubits - 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: Rot") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyRot({0}, false, {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyRot({0}, false,
                                           {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyRot({num_qubits - 1}, false,
                            {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyRot({num_qubits - 1}, false,
                                           {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("Rot", {0}, false,
                                  {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "Rot", {0}, false, {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("Rot", {num_qubits - 1}, false,
                                  {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "Rot", {num_qubits - 1}, false,
                        {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: IsingXX") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingXX({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingXX({0, 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingXX({0, num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingXX({0, num_qubits - 1}, false,
                                               {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingXX({num_qubits - 2, num_qubits - 1}, false,
                                {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingXX({num_qubits - 2, num_qubits - 1},
                                               false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingXX", {0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("IsingXX", {0, 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingXX", {0, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("IsingXX", {0, num_qubits - 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingXX", {num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "IsingXX", {num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: IsingYY") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingYY({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingYY({0, 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingYY({0, num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingYY({0, num_qubits - 1}, false,
                                               {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingYY({num_qubits - 2, num_qubits - 1}, false,
                                {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingYY({num_qubits - 2, num_qubits - 1},
                                               false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingYY", {0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("IsingYY", {0, 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingYY", {0, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("IsingYY", {0, num_qubits - 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingYY", {num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "IsingYY", {num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: IsingZZ") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingZZ({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingZZ({0, 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingZZ({0, num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingZZ({0, num_qubits - 1}, false,
                                               {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyIsingZZ({num_qubits - 2, num_qubits - 1}, false,
                                {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyIsingZZ({num_qubits - 2, num_qubits - 1},
                                               false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingZZ", {0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("IsingZZ", {0, 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingZZ", {0, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("IsingZZ", {0, num_qubits - 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("IsingZZ", {num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "IsingZZ", {num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: ControlledPhaseShift") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyControlledPhaseShift({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyControlledPhaseShift({0, 1}, false,
                                                            {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyControlledPhaseShift({0, num_qubits - 1}, false,
                                             {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyControlledPhaseShift({0, num_qubits - 1},
                                                            false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyControlledPhaseShift({num_qubits - 2, num_qubits - 1},
                                             false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyControlledPhaseShift(
                        {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("ControlledPhaseShift", {0, 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("ControlledPhaseShift", {0, 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("ControlledPhaseShift", {0, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("ControlledPhaseShift",
                                                 {0, num_qubits - 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("ControlledPhaseShift",
                                  {num_qubits - 2, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "ControlledPhaseShift",
                        {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: CRX") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRX({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRX({0, 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRX({0, num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRX({0, num_qubits - 1}, false,
                                           {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRX({num_qubits - 2, num_qubits - 1}, false,
                            {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRX({num_qubits - 2, num_qubits - 1},
                                           false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRX", {0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("CRX", {0, 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRX", {0, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("CRX", {0, num_qubits - 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRX", {num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "CRX", {num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: CRY") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRY({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRY({0, 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRY({0, num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRY({0, num_qubits - 1}, false,
                                           {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRY({num_qubits - 2, num_qubits - 1}, false,
                            {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRY({num_qubits - 2, num_qubits - 1},
                                           false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRY", {0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("CRY", {0, 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRY", {0, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("CRY", {0, num_qubits - 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRY", {num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "CRY", {num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: CRZ") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRZ({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRZ({0, 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRZ({0, num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRZ({0, num_qubits - 1}, false,
                                           {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRZ({num_qubits - 2, num_qubits - 1}, false,
                            {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRZ({num_qubits - 2, num_qubits - 1},
                                           false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRZ", {0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("CRZ", {0, 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRZ", {0, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("CRZ", {0, num_qubits - 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRZ", {num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "CRZ", {num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: CRot") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRot({0, 1}, false, {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRot({0, 1}, false,
                                            {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRot({0, num_qubits - 1}, false,
                             {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRot({0, num_qubits - 1}, false,
                                            {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCRot({num_qubits - 2, num_qubits - 1}, false,
                             {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyCRot({num_qubits - 2, num_qubits - 1},
                                            false,
                                            {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRot", {0, 1}, false,
                                  {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "CRot", {0, 1}, false,
                        {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRot", {0, num_qubits - 1}, false,
                                  {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "CRot", {0, num_qubits - 1}, false,
                        {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CRot", {num_qubits - 2, num_qubits - 1},
                                  false, {angles[0], angles[1], angles[2]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "CRot", {num_qubits - 2, num_qubits - 1}, false,
                        {angles[0], angles[1], angles[2]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: SingleExcitation") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitation({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitation({0, 1}, false,
                                                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitation({0, num_qubits - 1}, false,
                                         {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitation({0, num_qubits - 1},
                                                        false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitation({num_qubits - 2, num_qubits - 1},
                                         false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitation(
                        {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitation", {0, 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("SingleExcitation", {0, 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitation", {0, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("SingleExcitation",
                                                 {0, num_qubits - 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitation",
                                  {num_qubits - 2, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "SingleExcitation", {num_qubits - 2, num_qubits - 1},
                        false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: SingleExcitationMinus") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitationMinus({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitationMinus({0, 1}, false,
                                                             {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitationMinus({0, num_qubits - 1}, false,
                                              {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitationMinus(
                        {0, num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitationMinus({num_qubits - 2, num_qubits - 1},
                                              false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitationMinus(
                        {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitationMinus", {0, 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("SingleExcitationMinus",
                                                 {0, 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitationMinus", {0, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("SingleExcitationMinus",
                                                 {0, num_qubits - 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitationMinus",
                                  {num_qubits - 2, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "SingleExcitationMinus",
                        {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: SingleExcitationPlus") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitationPlus({0, 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitationPlus({0, 1}, false,
                                                            {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitationPlus({0, num_qubits - 1}, false,
                                             {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitationPlus({0, num_qubits - 1},
                                                            false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySingleExcitationPlus({num_qubits - 2, num_qubits - 1},
                                             false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applySingleExcitationPlus(
                        {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitationPlus", {0, 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("SingleExcitationPlus", {0, 1},
                                                 false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitationPlus", {0, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("SingleExcitationPlus",
                                                 {0, num_qubits - 1}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SingleExcitationPlus",
                                  {num_qubits - 2, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "SingleExcitationPlus",
                        {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: DoubleExcitation") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitation({0, 1, 2, 3}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitation({0, 1, 2, 3}, false,
                                                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitation({0, 1, num_qubits - 2, num_qubits - 1},
                                         false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitation(
                        {0, 1, num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitation({num_qubits - 4, num_qubits - 3,
                                          num_qubits - 2, num_qubits - 1},
                                         false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitation(
                        {num_qubits - 4, num_qubits - 3, num_qubits - 2,
                         num_qubits - 1},
                        false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitation", {0, 1, 2, 3}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "DoubleExcitation", {0, 1, 2, 3}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitation",
                                  {0, 1, num_qubits - 2, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "DoubleExcitation",
                        {0, 1, num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitation",
                                  {num_qubits - 4, num_qubits - 3,
                                   num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "DoubleExcitation",
                        {num_qubits - 4, num_qubits - 3, num_qubits - 2,
                         num_qubits - 1},
                        false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: DoubleExcitationMinus") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitationMinus({0, 1, 2, 3}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitationMinus(
                        {0, 1, 2, 3}, false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitationMinus(
                    {0, 1, num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitationMinus(
                        {0, 1, num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitationMinus({num_qubits - 4, num_qubits - 3,
                                               num_qubits - 2, num_qubits - 1},
                                              false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitationMinus(
                        {num_qubits - 4, num_qubits - 3, num_qubits - 2,
                         num_qubits - 1},
                        false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitationMinus", {0, 1, 2, 3}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("DoubleExcitationMinus",
                                                 {0, 1, 2, 3}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitationMinus",
                                  {0, 1, num_qubits - 2, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "DoubleExcitationMinus",
                        {0, 1, num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitationMinus",
                                  {num_qubits - 4, num_qubits - 3,
                                   num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "DoubleExcitationMinus",
                        {num_qubits - 4, num_qubits - 3, num_qubits - 2,
                         num_qubits - 1},
                        false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: DoubleExcitationPlus") {
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitationPlus({0, 1, 2, 3}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitationPlus({0, 1, 2, 3}, false,
                                                            {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitationPlus(
                    {0, 1, num_qubits - 2, num_qubits - 1}, false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitationPlus(
                        {0, 1, num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyDoubleExcitationPlus({num_qubits - 4, num_qubits - 3,
                                              num_qubits - 2, num_qubits - 1},
                                             false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyDoubleExcitationPlus(
                        {num_qubits - 4, num_qubits - 3, num_qubits - 2,
                         num_qubits - 1},
                        false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitationPlus", {0, 1, 2, 3}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation("DoubleExcitationPlus",
                                                 {0, 1, 2, 3}, false,
                                                 {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitationPlus",
                                  {0, 1, num_qubits - 2, num_qubits - 1}, false,
                                  {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "DoubleExcitationPlus",
                        {0, 1, num_qubits - 2, num_qubits - 1}, false,
                        {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                                nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("DoubleExcitationPlus",
                                  {num_qubits - 4, num_qubits - 3,
                                   num_qubits - 2, num_qubits - 1},
                                  false, {angles[0]});
                sv.CopyGpuDataToHost(local_state.data(),
                                     static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if (mpi_manager.getRank() == 0) {
                    svdat.cuda_sv.applyOperation(
                        "DoubleExcitationPlus",
                        {num_qubits - 4, num_qubits - 3, num_qubits - 2,
                         num_qubits - 1},
                        false, {angles[0]});
                    svdat.cuda_sv.CopyGpuDataToHost(
                        expected_sv.data(), static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }
}
*/
/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/
/*
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRX", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.1875866881010216, 0.04752321187375822},
                          {0.12208945659110043, 0.12517800600090637},
                          {0.292747949658552, 0.1981648805024429},
                          {0.16710253718877383, 0.1954570255012325},
                          {0.26464562086473914, 0.04386981831305299},
                          {0.2036051452704119, -0.02156519384027778},
                          {0.0254072177792504, 0.1571305624718667},
                          {0.24598704035093433, -0.018670488230388953},
                          {0.18639872920943937, 0.09292785397434067},
                          {0.04735521481393599, 0.23419044180146675},
                          {0.33789465554283776, 0.24908166072868462},
                          {0.21392239896831913, -0.019710116503746263},
                          {0.047685870888374726, 0.17427863024119702},
                          {0.1987790605074908, 0.20141955277753637},
                          {0.17299911217015343, 0.05318626021446178},
                          {0.19628847689417087, 0.18426669500353973}},

        std::vector<cp_t>{{0.18833122191631846, 0.06741247983417664},
                          {0.08828188999864703, 0.09660075459531156},
                          {0.27919553607063063, 0.22282025445920764},
                          {0.21626587729860575, 0.1822255055614605},
                          {0.22391331180061663, 0.01972931534801057},
                          {0.1667747107894936, -0.02666015881239122},
                          {0.018160648216742292, 0.14399019313584357},
                          {0.2378225892271685, 0.017437810245895316},
                          {0.22020629580189277, 0.12150510537993547},
                          {0.04661068099863915, 0.21430117384104833},
                          {0.28873131543300584, 0.26231318066845666},
                          {0.22747481255624047, -0.044365490460510984},
                          {0.08451630536929304, 0.17937359521331045},
                          {0.23951136957161334, 0.22556005574257879},
                          {0.18116356329391925, 0.017077961738177508},
                          {0.20353504645667897, 0.19740706433956287}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyRX({0}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyRX({num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("RX", {0}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("RX", {num_qubits - 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRY", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.12767100481299776, 0.05644317761580852},
                          {0.06739183865960663, 0.07903994995551314},
                          {0.17472513089843497, 0.19608149968837033},
                          {0.1304067622035783, 0.22663737319674956},
                          {0.21271318562329425, 0.007044096013596866},
                          {0.11679544360585836, -0.02926434831044669},
                          {-0.013434093656665199, 0.17325454810482382},
                          {0.16144355713430797, -0.025301808378209698},
                          {0.20264549116209435, 0.14239515671897587},
                          {0.03564137878027103, 0.274961390944369},
                          {0.33314318306829493, 0.3481722957634841},
                          {0.20073605778540315, 0.06010491471168469},
                          {0.08191211587123756, 0.2293528623969457},
                          {0.22682615023719963, 0.23676018191990117},
                          {0.13842394466981423, 0.09345699383103805},
                          {0.23279940142565922, 0.22900180621297037}},

        std::vector<cp_t>{{0.14756027277341618, 0.055698643800511684},
                          {0.10452865195130201, 0.14606805733994677},
                          {0.1993805048551997, 0.20963391327629166},
                          {0.2115144048240629, 0.2813161405962599},
                          {0.18857268265825183, 0.04777640507771939},
                          {0.20100095577235644, 0.02841407334335745},
                          {-0.026574462992688338, 0.18050111766733193},
                          {0.2032474217268293, 0.05770854386247158},
                          {0.16550867787039897, 0.07536704933454225},
                          {0.05553064674068945, 0.2742168571290722},
                          {0.25203554044781035, 0.2934935283639737},
                          {0.22539143174216789, 0.07365732829960603},
                          {-0.0022933962952605066, 0.17167444074314153},
                          {0.2026856472721572, 0.2774924909840237},
                          {0.09662008007729289, 0.010446641590356763},
                          {0.21965903208963608, 0.23624837577547847}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyRY({0}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyRY({num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("RY", {0}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("RY", {num_qubits - 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRZ", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.09791401219094054, 0.114929230389338},
                          {0.2846159036261283, 0.20896501819533683},
                          {0.21084550974310806, 0.1960807418253313},
                          {0.23069053568073156, 0.0067729362147416275},
                          {0.15999748690937868, -0.013703779679122279},
                          {0.0514715054362289, 0.18176542794818454},
                          {0.2045117320076819, -0.02033742456644565},
                          {0.14429060004046249, 0.16020271083802284},
                          {-0.029305014762791147, 0.25299877929913567},
                          {0.2259205020010718, 0.35422096098183514},
                          {0.16466399912430643, 0.047542289852867514},
                          {-0.0071172014685187066, 0.22648222528149717},
                          {0.1478778627338016, 0.2726686858107655},
                          {0.12421749871021645, 0.08351669180701665},
                          {0.1465889818729762, 0.263845794408402}},

        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.045429227014373304, 0.1439863434985753},
                          {0.2846159036261283, 0.20896501819533683},
                          {0.11784413734476112, 0.2627094318578463},
                          {0.23069053568073156, 0.0067729362147416275},
                          {0.1527039474967227, 0.04968393919295135},
                          {0.0514715054362289, 0.18176542794818454},
                          {0.19628754532973963, 0.06090861117447334},
                          {0.19528631758643603, 0.09136706180794868},
                          {-0.029305014762791147, 0.25299877929913567},
                          {0.3460267015752614, 0.2382815230357907},
                          {0.16466399912430643, 0.047542289852867514},
                          {0.08164095607238234, 0.21137551233950838},
                          {0.1478778627338016, 0.2726686858107655},
                          {0.14693482451317494, 0.02855139473814395},
                          {0.1465889818729762, 0.263845794408402}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyRZ({0}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyRZ({num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("RZ", {0}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("RZ", {num_qubits - 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyPhaseShift",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.10958702924257067, 0.18567543952196755},
                          {-0.07898396370748247, 0.2421336401538524},
                          {0.15104429198852115, 0.3920435999745404},
                          {0.15193648720587818, 0.07930829583090077},
                          {-0.05197040342070914, 0.22055368982062182},
                          {0.09075924552920023, 0.29661226181575395},
                          {0.10514921359740333, 0.10653012567371957},
                          {0.09124889440527104, 0.2877091797342799}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.015917996547459956, 0.1501415970580047},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.06330279338538422, 0.2808847497519412},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1397893602952355, 0.07903115931744624},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.18027418980248633, 0.09869080938889352},
                          {0.17324175995006008, 0.12834320562185453},
                          {-0.07898396370748247, 0.2421336401538524},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.15193648720587818, 0.07930829583090077},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.09075924552920023, 0.29661226181575395},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.09124889440527104, 0.2877091797342799}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyPhaseShift({0}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyPhaseShift({num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("PhaseShift", {0}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("PhaseShift", {num_qubits - 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRot", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<std::vector<PrecisionT>> angles{{0.4, 0.3, 0.2}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.15681012817121934, 0.008982433552624354},
                          {0.10825875361177284, 0.06534966881138937},
                          {0.2617644167773202, 0.12819203754862057},
                          {0.20159859304771044, 0.16748478541925405},
                          {0.2253085496099673, -0.04989076566943143},
                          {0.13000169633812203, -0.06761329857673015},
                          {0.04886570795802619, 0.16318061245716942},
                          {0.17346465653132614, -0.0772411332528724},
                          {0.15198238964112099, 0.18182009334518368},
                          {-0.04096093969063638, 0.26456513842651413},
                          {0.22650412271257353, 0.40611233404339736},
                          {0.18573422567641262, 0.09532911188246976},
                          {0.004841962975895488, 0.22656648739596288},
                          {0.14256581918009412, 0.2832067299864025},
                          {0.11890657056773607, 0.12222303074688182},
                          {0.14888656608406592, 0.27407700178656064}},
        std::vector<cp_t>{{0.17175191104797707, 0.009918765728752266},
                          {0.05632022889668771, 0.156107080015015},
                          {0.27919971361976087, 0.1401866473950276},
                          {0.12920853014584535, 0.30541194682630574},
                          {0.2041821323380667, -0.021217993121394878},
                          {0.17953120054123656, 0.06840312088235107},
                          {0.03848678812112871, 0.16761745054389973},
                          {0.19204303896856859, 0.10709469936852939},
                          {0.20173080131855414, 0.032556615426374606},
                          {-0.026130977192932347, 0.2625143322930839},
                          {0.338759826476869, 0.19565917738740204},
                          {0.20520493849776317, 0.1036207693512778},
                          {0.07513020150275973, 0.16156152395535361},
                          {0.12755692423653384, 0.31550512723628266},
                          {0.12164462285921541, -0.023459226158938577},
                          {0.139595997291366, 0.2804873715650301}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyRot({0}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyRot({num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("Rot", {0}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("Rot", {num_qubits - 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

// Two-qubit gates
// IsingXX Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyIsingXX",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.20646790809848536, 0.0743876799178009},
                          {0.11892604767001815, 0.08990251334676433},
                          {0.24405351277293644, 0.22865195094102878},
                          {0.20993222522615812, 0.190439005307186},
                          {0.2457644008672754, 0.01700535026901031},
                          {0.2067685541914942, 0.013710298813864255},
                          {0.07410165466486596, 0.12664349203328085},
                          {0.20315735231355, -0.01365246803634244},
                          {0.18021245239989217, 0.08113464775368001},
                          {0.024759591931980372, 0.2170251389200486},
                          {0.32339655234662434, 0.2934034589506944},
                          {0.1715338061081168, -0.027018789357948266},
                          {0.05387214769792193, 0.18607183646185768},
                          {0.22137468338944644, 0.21858485565895452},
                          {0.18749721536636682, 0.008864461992452007},
                          {0.2386770697543732, 0.19157536785774174}},
        std::vector<cp_t>{{0.2085895155272083, 0.048626691372537806},
                          {0.12359267335732806, 0.08228820566382836},
                          {0.2589372424597408, 0.24160604292084648},
                          {0.1809550939399247, 0.1965380544929437},
                          {0.22437863543325978, 0.010800156913587891},
                          {0.19377366776150842, 0.015142296698796012},
                          {0.017695324584099153, 0.15291935157026626},
                          {0.2108236322551537, -0.024364645265291914},
                          {0.1725461724582885, 0.09184682498262947},
                          {0.08116592201274717, 0.19074927938306319},
                          {0.33639143877661015, 0.29197146106576266},
                          {0.19291957154213243, -0.020813596002525848},
                          {0.08284927898415535, 0.17997278727609997},
                          {0.20649095370264206, 0.20563076367913682},
                          {0.1828305896790569, 0.016478769675387986},
                          {0.23655546232565025, 0.21733635640300483}},
        std::vector<cp_t>{{0.19951177988649257, 0.07909325333067677},
                          {0.07578371294162806, 0.08883476907349716},
                          {0.2431188434579001, 0.21148436090615938},
                          {0.16799159325026258, 0.197701227405552},
                          {0.2721875958489598, 0.004253593503919087},
                          {0.20285140340222416, -0.015324265259342953},
                          {0.030658825273761275, 0.15175617865765798},
                          {0.2266420312569944, 0.005757036749395192},
                          {0.1811471217149285, 0.09830223778854941},
                          {0.06670022390787592, 0.2097629168216826},
                          {0.3303526805586171, 0.2886978855378185},
                          {0.21467614083650688, -0.02595104508468109},
                          {0.09731497708902662, 0.16095914983748055},
                          {0.19789000444600205, 0.1991753508732169},
                          {0.16107402038468246, 0.02161621875754323},
                          {0.24259422054364324, 0.22060993193094894}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingXX({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingXX({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingXX({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingXX", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingXX", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingXX", {1, num_qubits - 2}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// IsingYY Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyIsingYY",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.11770975055758431, 0.08949439285978969},
                          {0.024417191535295688, 0.16901306054114898},
                          {0.22133618696997795, 0.2836172480099015},
                          {0.11875742186171105, 0.2683511683759916},
                          {0.2457644008672754, 0.01700535026901031},
                          {0.2067685541914942, 0.013710298813864255},
                          {0.07410165466486596, 0.12664349203328085},
                          {0.20315735231355, -0.01365246803634244},
                          {0.18021245239989217, 0.08113464775368001},
                          {0.024759591931980372, 0.2170251389200486},
                          {0.32339655234662434, 0.2934034589506944},
                          {0.1715338061081168, -0.027018789357948266},
                          {0.020651606905941693, 0.2517859011591479},
                          {0.1688898982128792, 0.24764196876819183},
                          {0.08365510785702455, 0.1032036245527086},
                          {0.14567569735602626, 0.2582040578902567}},
        std::vector<cp_t>{{0.11558814312886137, 0.11525538140505279},
                          {0.12359267335732806, 0.08228820566382836},
                          {0.2589372424597408, 0.24160604292084648},
                          {0.14773455314794448, 0.2622521191902339},
                          {0.2161544487553175, 0.09204619265450688},
                          {0.19377366776150842, 0.015142296698796012},
                          {0.017695324584099153, 0.15291935157026626},
                          {0.18997564508226786, 0.0649358318733196},
                          {0.16703074516861002, 0.15972294766334205},
                          {0.08116592201274717, 0.19074927938306319},
                          {0.33639143877661015, 0.29197146106576266},
                          {0.1419238539961589, 0.048022053027548306},
                          {-0.00832552438029173, 0.25788495034490555},
                          {0.20649095370264206, 0.20563076367913682},
                          {0.1828305896790569, 0.016478769675387986},
                          {0.1477973047847492, 0.23244306934499362}},
        std::vector<cp_t>{{0.1246658787695771, 0.08478881944691383},
                          {0.06755952626368578, 0.17008080481441615},
                          {0.2431188434579001, 0.21148436090615938},
                          {0.16799159325026258, 0.197701227405552},
                          {0.2721875958489598, 0.004253593503919087},
                          {0.20285140340222416, -0.015324265259342953},
                          {-0.0025617155182189634, 0.21747024335494816},
                          {0.17415724608042715, 0.034814149858632494},
                          {0.15842979591197, 0.15326753485742212},
                          {-0.024474579456571166, 0.2876750798904882},
                          {0.3303526805586171, 0.2886978855378185},
                          {0.21467614083650688, -0.02595104508468109},
                          {0.09731497708902662, 0.16095914983748055},
                          {0.19789000444600205, 0.1991753508732169},
                          {0.11007830283870892, 0.09045186778761738},
                          {0.1417585465667562, 0.2291694938170495}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingYY({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingYY({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingYY({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingYY", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingYY", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingYY", {1, num_qubits - 2}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// IsingZZ Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyIsingZZ",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.09791401219094054, 0.114929230389338},
                          {0.2846159036261283, 0.20896501819533683},
                          {0.21084550974310806, 0.1960807418253313},
                          {0.20984254850784573, 0.09607341335335315},
                          {0.1527039474967227, 0.04968393919295135},
                          {-0.023374395680686586, 0.1874609940644216},
                          {0.19628754532973963, 0.06090861117447334},
                          {0.14429060004046249, 0.16020271083802284},
                          {-0.029305014762791147, 0.25299877929913567},
                          {0.2259205020010718, 0.35422096098183514},
                          {0.16466399912430643, 0.047542289852867514},
                          {0.08164095607238234, 0.21137551233950838},
                          {0.24238671886852403, 0.19355813861638085},
                          {0.14693482451317494, 0.02855139473814395},
                          {0.23776378523742325, 0.18593363133959642}},
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.045429227014373304, 0.1439863434985753},
                          {0.18077379611678607, 0.3033041807555934},
                          {0.21084550974310806, 0.1960807418253313},
                          {0.23069053568073156, 0.0067729362147416275},
                          {0.1527039474967227, 0.04968393919295135},
                          {-0.023374395680686586, 0.1874609940644216},
                          {0.2045117320076819, -0.02033742456644565},
                          {0.19528631758643603, 0.09136706180794868},
                          {-0.029305014762791147, 0.25299877929913567},
                          {0.2259205020010718, 0.35422096098183514},
                          {0.1701794264139849, -0.020333832827845056},
                          {0.08164095607238234, 0.21137551233950838},
                          {0.1478778627338016, 0.2726686858107655},
                          {0.12421749871021645, 0.08351669180701665},
                          {0.23776378523742325, 0.18593363133959642}},
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.09791401219094054, 0.114929230389338},
                          {0.18077379611678607, 0.3033041807555934},
                          {0.11784413734476112, 0.2627094318578463},
                          {0.20984254850784573, 0.09607341335335315},
                          {0.1527039474967227, 0.04968393919295135},
                          {0.0514715054362289, 0.18176542794818454},
                          {0.2045117320076819, -0.02033742456644565},
                          {0.19528631758643603, 0.09136706180794868},
                          {0.0715306592140959, 0.24443921741303512},
                          {0.2259205020010718, 0.35422096098183514},
                          {0.16466399912430643, 0.047542289852867514},
                          {-0.0071172014685187066, 0.22648222528149717},
                          {0.1478778627338016, 0.2726686858107655},
                          {0.14693482451317494, 0.02855139473814395},
                          {0.23776378523742325, 0.18593363133959642}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingZZ({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingZZ({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyIsingZZ({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingZZ", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingZZ", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("IsingZZ", {1, num_qubits - 2}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// ControlledPhaseShift Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyControlledPhaseShift",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {-0.05197040342070914, 0.22055368982062182},
                          {0.09075924552920023, 0.29661226181575395},
                          {0.10514921359740333, 0.10653012567371957},
                          {0.09124889440527104, 0.2877091797342799}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.06330279338538422, 0.2808847497519412},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.18027418980248633, 0.09869080938889352},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.15193648720587818, 0.07930829583090077},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.09124889440527104, 0.2877091797342799}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {-0.06015121422483318, 0.179080479383814},
                          {0.18027418980248633, 0.09869080938889352},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.10514921359740333, 0.10653012567371957},
                          {0.09124889440527104, 0.2877091797342799}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyControlledPhaseShift({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyControlledPhaseShift({num_qubits - 2, num_qubits - 1}, false,
                                     angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyControlledPhaseShift({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("ControlledPhaseShift", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("ControlledPhaseShift",
                          {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("ControlledPhaseShift", {1, num_qubits - 2}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// CRX Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyCRX", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.21416753758389978, 0.11823152985199137},
                          {0.0683672502930136, 0.20916372475889308},
                          {0.29733226468964585, 0.26876859347437654},
                          {0.21300911445136922, -0.02535185302189157},
                          {0.06275973607491858, 0.1845110442954657},
                          {0.24555012778960633, 0.2288336312705229},
                          {0.1956292613987905, -0.0019356757004419106},
                          {0.19493409720003896, 0.19095165153364294}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.27919553607063063, 0.22282025445920764},
                          {0.21626587729860575, 0.1822255055614605},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.018160648216742292, 0.14399019313584357},
                          {0.2378225892271685, 0.017437810245895316},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.28873131543300584, 0.26231318066845666},
                          {0.22747481255624047, -0.044365490460510984},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.18116356329391925, 0.017077961738177508},
                          {0.20353504645667897, 0.19740706433956287}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2576894926527464, 0.04857539172592886},
                          {0.16046281054202183, -0.022632938113544956},
                          {0.024472548464214074, 0.13996297243699732},
                          {0.20404640837503876, -0.011408266132022973},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.048620540203411056, 0.19144622027606642},
                          {0.24071969248338637, 0.19415733067917038},
                          {0.17995524038214622, 0.04848068680158591},
                          {0.23943081162256097, 0.1853344392768069}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRX({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRX({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRX({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRX", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRX", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRX", {1, num_qubits - 2}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// CRY Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyCRY", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.16223510234245486, 0.08140580755253524},
                          {-0.018442451371539943, 0.20146457028872417},
                          {0.2584909532537303, 0.2848925791073337},
                          {0.12846563123474286, -0.03198317316971232},
                          {0.0716797018169689, 0.24442672758348954},
                          {0.1994120717442131, 0.2835312492020167},
                          {0.19354588058471792, 0.1160871430596751},
                          {0.22611444489555602, 0.22764742651883846}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.1993805048551997, 0.20963391327629166},
                          {0.2115144048240629, 0.2813161405962599},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {-0.026574462992688338, 0.18050111766733193},
                          {0.2032474217268293, 0.05770854386247158},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.25203554044781035, 0.2934935283639737},
                          {0.22539143174216789, 0.07365732829960603},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.09662008007729289, 0.010446641590356763},
                          {0.21965903208963608, 0.23624837577547847}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2174187590361701, 0.014000224225589643},
                          {0.1157276993325912, 0.013877986417943402},
                          {0.058698793447076916, 0.19503720459274598},
                          {0.23209349810474758, 0.02393236301034184},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.009779228767495457, 0.20757020590902353},
                          {0.15617620926676, 0.18752601053134965},
                          {0.1431295180826901, 0.10041312204303082},
                          {0.23173165715239205, 0.27214414094136047}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRY({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRY({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRY({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRY", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRY", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRY", {1, num_qubits - 2}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}
// CRZ Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyCRZ", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.19528631758643603, 0.09136706180794868},
                          {0.0715306592140959, 0.24443921741303512},
                          {0.3460267015752614, 0.2382815230357907},
                          {0.1701794264139849, -0.020333832827845056},
                          {-0.0071172014685187066, 0.22648222528149717},
                          {0.1478778627338016, 0.2726686858107655},
                          {0.12421749871021645, 0.08351669180701665},
                          {0.1465889818729762, 0.263845794408402}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.2846159036261283, 0.20896501819533683},
                          {0.11784413734476112, 0.2627094318578463},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.0514715054362289, 0.18176542794818454},
                          {0.19628754532973963, 0.06090861117447334},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.3460267015752614, 0.2382815230357907},
                          {0.16466399912430643, 0.047542289852867514},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.14693482451317494, 0.02855139473814395},
                          {0.1465889818729762, 0.263845794408402}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.23069053568073156, 0.0067729362147416275},
                          {0.15999748690937868, -0.013703779679122279},
                          {-0.023374395680686586, 0.1874609940644216},
                          {0.19628754532973963, 0.06090861117447334},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.08164095607238234, 0.21137551233950838},
                          {0.24238671886852403, 0.19355813861638085},
                          {0.12421749871021645, 0.08351669180701665},
                          {0.1465889818729762, 0.263845794408402}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRZ({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRZ({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRZ({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRZ", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRZ", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRZ", {1, num_qubits - 2}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// CRot Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyCRot",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<std::vector<PrecisionT>> angles{{0.4, 0.3, 0.2}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.19882725898288345, 0.0368304215190272},
                          {0.06844717210134421, 0.19508864487497662},
                          {0.34423718069125153, 0.18970666921007756},
                          {0.13968781282676393, -0.07384857547711987},
                          {-0.0016843985305852564, 0.23861672758372268},
                          {0.12556033363237104, 0.32027058656062857},
                          {0.16186128437918007, 0.13502094254448443},
                          {0.143781588275739, 0.27356533494980884}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.27919971361976087, 0.1401866473950276},
                          {0.12920853014584535, 0.30541194682630574},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.03848678812112871, 0.16761745054389973},
                          {0.19204303896856859, 0.10709469936852939},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.338759826476869, 0.19565917738740204},
                          {0.20520493849776317, 0.1036207693512778},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.12164462285921541, -0.023459226158938577},
                          {0.139595997291366, 0.2804873715650301}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.22830801184905478, -0.04433117752949302},
                          {0.1259628163808648, -0.03540414168066641},
                          {-0.007300495018646443, 0.18657139058160394},
                          {0.21109612625941312, 0.07964916748538081},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.08147024138239997, 0.18933384276468704},
                          {0.2318413989669087, 0.1294572948552116},
                          {0.1229507629150203, 0.12707589611361345},
                          {0.15132716647205935, 0.30644652153910734}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRot({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRot({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyCRot({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRot", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRot", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("CRot", {1, num_qubits - 2}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// SingleExcitation Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applySingleExcitation",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.18584871757925156, 0.025925316011060608},
                          {0.1520709362600004, -0.032427757231528985},
                          {-0.043921164095251056, 0.12456011121920826},
                          {0.16646157732835448, 0.017527879659174597},
                          {0.214438697382755, 0.13620887990942868},
                          {0.05280668166168919, 0.2523657680624134},
                          {0.2888213848462851, 0.33367419256727066},
                          {0.20804473063960516, 0.017716321851482364},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.02450203832252862, 0.07753673318928551},
                          {0.24722340642607585, 0.2823769920637487},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.15350293414493216, -0.019432870801543205},
                          {0.045742414313807975, 0.18825998071263106},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {-0.03685689674736984, 0.1886658985689906},
                          {0.29025338273121687, 0.34666907899725646},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.16764964226672646, 0.22175474931209394},
                          {0.17513143520888802, 0.10328847133994154},
                          {0.1960850292216881, 0.22946370987301284}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.18804461130215144, 0.24571060588902222},
                          {0.13265096410789778, 0.2257483171352608},
                          {0.2674361233744169, 0.10334422853871852},
                          {0.18966506221930818, 0.064490765956088},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2784202453171722, 0.2518721632383624},
                          {0.12786643917195334, -0.033650199554850005},
                          {0.09523159627495403, 0.27898196859759755},
                          {0.2290703521415191, 0.2358711258584124},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitation({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitation({num_qubits - 2, num_qubits - 1}, false,
                                 angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitation({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitation", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitation", {num_qubits - 2, num_qubits - 1},
                          false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitation", {1, num_qubits - 2}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}
// SingleExcitationMinus Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applySingleExcitationMinus",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.09791401219094054, 0.114929230389338},
                          {0.2846159036261283, 0.20896501819533683},
                          {0.21084550974310806, 0.1960807418253313},
                          {0.18584871757925156, 0.025925316011060608},
                          {0.1520709362600004, -0.032427757231528985},
                          {-0.043921164095251056, 0.12456011121920826},
                          {0.16646157732835448, 0.017527879659174597},
                          {0.214438697382755, 0.13620887990942868},
                          {0.05280668166168919, 0.2523657680624134},
                          {0.2888213848462851, 0.33367419256727066},
                          {0.20804473063960516, 0.017716321851482364},
                          {0.08164095607238234, 0.21137551233950838},
                          {0.24238671886852403, 0.19355813861638085},
                          {0.14693482451317494, 0.02855139473814395},
                          {0.23776378523742325, 0.18593363133959642}},
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.02450203832252862, 0.07753673318928551},
                          {0.24722340642607585, 0.2823769920637487},
                          {0.21084550974310806, 0.1960807418253313},
                          {0.23069053568073156, 0.0067729362147416275},
                          {0.15350293414493216, -0.019432870801543205},
                          {0.045742414313807975, 0.18825998071263106},
                          {0.2045117320076819, -0.02033742456644565},
                          {0.19528631758643603, 0.09136706180794868},
                          {-0.03685689674736984, 0.1886658985689906},
                          {0.29025338273121687, 0.34666907899725646},
                          {0.1701794264139849, -0.020333832827845056},
                          {0.08164095607238234, 0.21137551233950838},
                          {0.16764964226672646, 0.22175474931209394},
                          {0.17513143520888802, 0.10328847133994154},
                          {0.23776378523742325, 0.18593363133959642}},
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.09791401219094054, 0.114929230389338},
                          {0.18804461130215144, 0.24571060588902222},
                          {0.13265096410789778, 0.2257483171352608},
                          {0.2674361233744169, 0.10334422853871852},
                          {0.18966506221930818, 0.064490765956088},
                          {0.0514715054362289, 0.18176542794818454},
                          {0.2045117320076819, -0.02033742456644565},
                          {0.19528631758643603, 0.09136706180794868},
                          {0.0715306592140959, 0.24443921741303512},
                          {0.2784202453171722, 0.2518721632383624},
                          {0.12786643917195334, -0.033650199554850005},
                          {0.09523159627495403, 0.27898196859759755},
                          {0.2290703521415191, 0.2358711258584124},
                          {0.14693482451317494, 0.02855139473814395},
                          {0.23776378523742325, 0.18593363133959642}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitationMinus({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitationMinus({num_qubits - 2, num_qubits - 1}, false,
                                      angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitationMinus({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitationMinus", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitationMinus",
                          {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitationMinus", {1, num_qubits - 2}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// SingleExcitationPlus Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applySingleExcitationPlus",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.14547855893204473, 0.11479806873744039},
                          {0.045429227014373304, 0.1439863434985753},
                          {0.18077379611678607, 0.3033041807555934},
                          {0.11784413734476112, 0.2627094318578463},
                          {0.18584871757925156, 0.025925316011060608},
                          {0.1520709362600004, -0.032427757231528985},
                          {-0.043921164095251056, 0.12456011121920826},
                          {0.16646157732835448, 0.017527879659174597},
                          {0.214438697382755, 0.13620887990942868},
                          {0.05280668166168919, 0.2523657680624134},
                          {0.2888213848462851, 0.33367419256727066},
                          {0.20804473063960516, 0.017716321851482364},
                          {-0.0071172014685187066, 0.22648222528149717},
                          {0.1478778627338016, 0.2726686858107655},
                          {0.12421749871021645, 0.08351669180701665},
                          {0.1465889818729762, 0.263845794408402}},
        std::vector<cp_t>{{0.14547855893204473, 0.11479806873744039},
                          {0.02450203832252862, 0.07753673318928551},
                          {0.24722340642607585, 0.2823769920637487},
                          {0.11784413734476112, 0.2627094318578463},
                          {0.20984254850784573, 0.09607341335335315},
                          {0.15350293414493216, -0.019432870801543205},
                          {0.045742414313807975, 0.18825998071263106},
                          {0.19628754532973963, 0.06090861117447334},
                          {0.14429060004046249, 0.16020271083802284},
                          {-0.03685689674736984, 0.1886658985689906},
                          {0.29025338273121687, 0.34666907899725646},
                          {0.16466399912430643, 0.047542289852867514},
                          {-0.0071172014685187066, 0.22648222528149717},
                          {0.16764964226672646, 0.22175474931209394},
                          {0.17513143520888802, 0.10328847133994154},
                          {0.1465889818729762, 0.263845794408402}},
        std::vector<cp_t>{{0.14547855893204473, 0.11479806873744039},
                          {0.045429227014373304, 0.1439863434985753},
                          {0.18804461130215144, 0.24571060588902222},
                          {0.13265096410789778, 0.2257483171352608},
                          {0.2674361233744169, 0.10334422853871852},
                          {0.18966506221930818, 0.064490765956088},
                          {-0.023374395680686586, 0.1874609940644216},
                          {0.19628754532973963, 0.06090861117447334},
                          {0.14429060004046249, 0.16020271083802284},
                          {-0.029305014762791147, 0.25299877929913567},
                          {0.2784202453171722, 0.2518721632383624},
                          {0.12786643917195334, -0.033650199554850005},
                          {0.09523159627495403, 0.27898196859759755},
                          {0.2290703521415191, 0.2358711258584124},
                          {0.12421749871021645, 0.08351669180701665},
                          {0.1465889818729762, 0.263845794408402}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitationPlus({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitationPlus({num_qubits - 2, num_qubits - 1}, false,
                                     angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applySingleExcitationPlus({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitationPlus", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitationPlus",
                          {num_qubits - 2, num_qubits - 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("SingleExcitationPlus", {1, num_qubits - 2}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// DoubleExcitation Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyDoubleExcitation",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 6;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.14836184869237898, 0.06928854101488939},
                          {0.07094825327558218, 0.061139592727065234},
                          {0.10679034746239055, 0.12478353692681407},
                          {0.11571545213303741, 0.06594226247249863},
                          {0.1190627426132508, 0.07484925755324358},
                          {0.00906902232517842, 0.1237218622207207},
                          {0.1305100520050695, 0.11711448904556603},
                          {0.05088581583296845, 0.08008722296156004},
                          {0.07316105587189087, 0.06352039635890594},
                          {0.11095039687205051, 0.03583892505512524},
                          {0.03570968538901324, 0.09638794194868609},
                          {0.0713600372820485, 0.056471298297101645},
                          {0.14628197548279212, 0.005040172811679685},
                          {0.070043404380343, 0.07117993551880243},
                          {0.06074339783032044, 0.12942393723357834},
                          {0.13268568980573536, 0.08036089556546508},
                          {0.036258177794672615, 0.08363767253314282},
                          {0.13825718318791222, 0.07482394855854033},
                          {0.07835567694876108, 0.022806916071867203},
                          {0.12820641894084636, 0.1050400222703431},
                          {0.07756072170440836, 0.020371250936538677},
                          {0.08014481803707861, 0.06974261931716805},
                          {0.10823246498880147, 0.11082881021426993},
                          {0.0233312644662286, 0.028272496102782756},
                          {0.008840824821587321, 0.1265209337189973},
                          {0.02617377406378093, 0.05258049528203285},
                          {0.0508392238465671, 0.01130963009263731},
                          {0.11457119366932077, 0.03413193019742667},
                          {0.12112734694895362, 0.0186002687068472},
                          {0.09252244815846217, 0.02966503116247933},
                          {0.1437401906964705, 0.10570310943862452},
                          {0.03392060130372184, 0.04250381993713222},
                          {0.08626107014172368, 0.06021136873328912},
                          {0.01166779452451774, 0.03645181951160576},
                          {0.1119077793976233, 0.10963382996954998},
                          {0.13350973802938118, 0.005648970975886418},
                          {0.05996197257704452, 0.1297269037363724},
                          {0.05212466979002961, 0.08441484000268347},
                          {0.015560963020448076, 0.09045774083745448},
                          {0.0736093207875478, 0.04994035230603118},
                          {0.040527605427232036, 0.12006575976853436},
                          {0.0026016539235101596, 0.001501879850125351},
                          {0.08752150598418447, 0.0714082719633339},
                          {0.0649944068764156, 0.04190325980719882},
                          {0.09601550206852992, 0.13632483092843833},
                          {0.0503907955404767, 0.08748233712665994},
                          {0.06295764772507972, 0.14904685877449778},
                          {0.12316740102643296, 0.15182199339140806},
                          {0.06305718075195604, 0.00650192311401943},
                          {0.11148843236383083, 0.08934571675171843},
                          {0.11139801671381279, 0.014103743906341755},
                          {0.114804641395206, 0.10467054960473593},
                          {0.11460076196121058, 0.08524887644475407},
                          {0.13378710190861637, 0.1324168558193471},
                          {0.017717280830810585, 0.006437986898650014},
                          {0.11450190282689218, 0.02657063872874804},
                          {0.029424876402037975, 0.1520554860726633},
                          {0.02981337874417589, 0.06819108868747294},
                          {0.07902126685601678, 0.13001384412451206},
                          {0.057497920365339114, 0.00442340419954388},
                          {0.11496868516821211, 0.04572193372968664},
                          {0.08564011282247964, 0.06727215306155458},
                          {0.15373243881038598, 0.15302409187671534},
                          {0.1381066836079441, 0.042936150447173645}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.14836184869237898, 0.06928854101488939},
                          {0.07094825327558218, 0.061139592727065234},
                          {0.10679034746239055, 0.12478353692681407},
                          {0.11571545213303741, 0.06594226247249863},
                          {0.1190627426132508, 0.07484925755324358},
                          {0.00906902232517842, 0.1237218622207207},
                          {0.1305100520050695, 0.11711448904556603},
                          {0.05088581583296845, 0.08008722296156004},
                          {0.07316105587189087, 0.06352039635890594},
                          {0.11095039687205051, 0.03583892505512524},
                          {0.03570968538901324, 0.09638794194868609},
                          {0.0713600372820485, 0.056471298297101645},
                          {0.13083854720946214, 0.003647972205328197},
                          {0.04649786738228028, 0.05201082205840395},
                          {0.03740120460558103, 0.12404209389167492},
                          {0.107232648658259, 0.05796419986517863},
                          {0.036258177794672615, 0.08363767253314282},
                          {0.13825718318791222, 0.07482394855854033},
                          {0.07835567694876108, 0.022806916071867203},
                          {0.12820641894084636, 0.1050400222703431},
                          {0.07756072170440836, 0.020371250936538677},
                          {0.08014481803707861, 0.06974261931716805},
                          {0.10823246498880147, 0.11082881021426993},
                          {0.0233312644662286, 0.028272496102782756},
                          {0.008840824821587321, 0.1265209337189973},
                          {0.02617377406378093, 0.05258049528203285},
                          {0.0508392238465671, 0.01130963009263731},
                          {0.11457119366932077, 0.03413193019742667},
                          {0.12112734694895362, 0.0186002687068472},
                          {0.09252244815846217, 0.02966503116247933},
                          {0.1437401906964705, 0.10570310943862452},
                          {0.03392060130372184, 0.04250381993713222},
                          {0.08626107014172368, 0.06021136873328912},
                          {0.01166779452451774, 0.03645181951160576},
                          {0.1119077793976233, 0.10963382996954998},
                          {0.13350973802938118, 0.005648970975886418},
                          {0.05996197257704452, 0.1297269037363724},
                          {0.05212466979002961, 0.08441484000268347},
                          {0.015560963020448076, 0.09045774083745448},
                          {0.0736093207875478, 0.04994035230603118},
                          {0.040527605427232036, 0.12006575976853436},
                          {0.0026016539235101596, 0.001501879850125351},
                          {0.08752150598418447, 0.0714082719633339},
                          {0.0649944068764156, 0.04190325980719882},
                          {0.09601550206852992, 0.13632483092843833},
                          {0.0503907955404767, 0.08748233712665994},
                          {0.06295764772507972, 0.14904685877449778},
                          {0.12316740102643296, 0.15182199339140806},
                          {0.09086197752443204, 0.007373645295331757},
                          {0.12318156265055501, 0.10170602101718566},
                          {0.12124532321617587, 0.03953517502409465},
                          {0.138876769212277, 0.11854935269595822},
                          {0.11460076196121058, 0.08524887644475407},
                          {0.13378710190861637, 0.1324168558193471},
                          {0.017717280830810585, 0.006437986898650014},
                          {0.11450190282689218, 0.02657063872874804},
                          {0.029424876402037975, 0.1520554860726633},
                          {0.02981337874417589, 0.06819108868747294},
                          {0.07902126685601678, 0.13001384412451206},
                          {0.057497920365339114, 0.00442340419954388},
                          {0.11496868516821211, 0.04572193372968664},
                          {0.08564011282247964, 0.06727215306155458},
                          {0.15373243881038598, 0.15302409187671534},
                          {0.1381066836079441, 0.042936150447173645}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyDoubleExcitation({0, 1, 2, 3}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("DoubleExcitation", {0, 1, 2, 3}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }
}

// DoubleExcitationMinus Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyDoubleExcitationMinus",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 6;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.14836184869237898, 0.06928854101488939},
                          {0.07094825327558218, 0.061139592727065234},
                          {0.10679034746239055, 0.12478353692681407},
                          {0.11571545213303741, 0.06594226247249863},
                          {0.1190627426132508, 0.07484925755324358},
                          {0.00906902232517842, 0.1237218622207207},
                          {0.1305100520050695, 0.11711448904556603},
                          {0.05088581583296845, 0.08008722296156004},
                          {0.07316105587189087, 0.06352039635890594},
                          {0.11095039687205051, 0.03583892505512524},
                          {0.03570968538901324, 0.09638794194868609},
                          {0.0713600372820485, 0.056471298297101645},
                          {0.14628197548279212, 0.005040172811679685},
                          {0.070043404380343, 0.07117993551880243},
                          {0.06074339783032044, 0.12942393723357834},
                          {0.13268568980573536, 0.08036089556546508},
                          {0.036258177794672615, 0.08363767253314282},
                          {0.13825718318791222, 0.07482394855854033},
                          {0.07835567694876108, 0.022806916071867203},
                          {0.12820641894084636, 0.1050400222703431},
                          {0.07756072170440836, 0.020371250936538677},
                          {0.08014481803707861, 0.06974261931716805},
                          {0.10823246498880147, 0.11082881021426993},
                          {0.0233312644662286, 0.028272496102782756},
                          {0.008840824821587321, 0.1265209337189973},
                          {0.02617377406378093, 0.05258049528203285},
                          {0.0508392238465671, 0.01130963009263731},
                          {0.11457119366932077, 0.03413193019742667},
                          {0.12112734694895362, 0.0186002687068472},
                          {0.09252244815846217, 0.02966503116247933},
                          {0.1437401906964705, 0.10570310943862452},
                          {0.03392060130372184, 0.04250381993713222},
                          {0.08626107014172368, 0.06021136873328912},
                          {0.01166779452451774, 0.03645181951160576},
                          {0.1119077793976233, 0.10963382996954998},
                          {0.13350973802938118, 0.005648970975886418},
                          {0.05996197257704452, 0.1297269037363724},
                          {0.05212466979002961, 0.08441484000268347},
                          {0.015560963020448076, 0.09045774083745448},
                          {0.0736093207875478, 0.04994035230603118},
                          {0.040527605427232036, 0.12006575976853436},
                          {0.0026016539235101596, 0.001501879850125351},
                          {0.08752150598418447, 0.0714082719633339},
                          {0.0649944068764156, 0.04190325980719882},
                          {0.09601550206852992, 0.13632483092843833},
                          {0.0503907955404767, 0.08748233712665994},
                          {0.06295764772507972, 0.14904685877449778},
                          {0.12316740102643296, 0.15182199339140806},
                          {0.06305718075195604, 0.00650192311401943},
                          {0.11148843236383083, 0.08934571675171843},
                          {0.11139801671381279, 0.014103743906341755},
                          {0.114804641395206, 0.10467054960473593},
                          {0.11460076196121058, 0.08524887644475407},
                          {0.13378710190861637, 0.1324168558193471},
                          {0.017717280830810585, 0.006437986898650014},
                          {0.11450190282689218, 0.02657063872874804},
                          {0.029424876402037975, 0.1520554860726633},
                          {0.02981337874417589, 0.06819108868747294},
                          {0.07902126685601678, 0.13001384412451206},
                          {0.057497920365339114, 0.00442340419954388},
                          {0.11496868516821211, 0.04572193372968664},
                          {0.08564011282247964, 0.06727215306155458},
                          {0.15373243881038598, 0.15302409187671534},
                          {0.1381066836079441, 0.042936150447173645}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.15916999740533416, 0.03843243408084209},
                          {0.08168057376378213, 0.04582562941528366},
                          {0.1294523121594332, 0.10108020714106368},
                          {0.12650955233190156, 0.04163869607861218},
                          {0.13155966660997137, 0.04970314030656599},
                          {0.03346798524671278, 0.11945392551449867},
                          {0.15117559720757903, 0.08885165180063768},
                          {0.06578236237508506, 0.06838135955824809},
                          {0.08432226029607685, 0.047719359472263564},
                          {0.11585887102963423, 0.013082091534268593},
                          {0.05414719707865628, 0.0873721811116716},
                          {0.08115670257545414, 0.04116858121595715},
                          {0.13083854720946214, 0.003647972205328197},
                          {0.04649786738228028, 0.05201082205840395},
                          {0.03740120460558103, 0.12404209389167492},
                          {0.107232648658259, 0.05796419986517863},
                          {0.052151668661400044, 0.07476709957984773},
                          {0.15036646817651608, 0.04586498914278501},
                          {0.08132481491510313, 0.006785426282262933},
                          {0.14651905719978323, 0.07747553170825378},
                          {0.08006181388675683, 0.00455624551472186},
                          {0.09240297705294515, 0.05243009287774301},
                          {0.1280933071309785, 0.0871171413632909},
                          {0.028483070402771454, 0.023073721804375908},
                          {0.03380042616170648, 0.12224253778427513},
                          {0.03609817298643397, 0.046332459894627004},
                          {0.0520727007774538, 0.0009839958818035704},
                          {0.11906836542974383, 0.010689781649032783},
                          {0.1224081673438474, -0.005834787260882545},
                          {0.0965716910292221, 0.010692332713802483},
                          {0.161874922809266, 0.0750393172406687},
                          {0.041688653101199666, 0.03491759018978901},
                          {0.09650374414721967, 0.041873721022849725},
                          {0.018677074039231953, 0.03340717707478584},
                          {0.1314579540205913, 0.08521580891019825},
                          {0.13197070934239635, -0.020987922656259594},
                          {0.08453948241561483, 0.11522839763381346},
                          {0.06785628651472436, 0.0723765900953472},
                          {0.03322195861277482, 0.08556312239201481},
                          {0.08206365149377894, 0.03432095567961683},
                          {0.06357313570378274, 0.10962084604330162},
                          {0.002848171522300832, 0.0009550734010969608},
                          {0.0999635364616187, 0.05259702170860746},
                          {0.07202373851021766, 0.028155589120074386},
                          {0.12118514746331548, 0.11453207498091375},
                          {0.06676639191339417, 0.07572740914102215},
                          {0.09131372604468878, 0.13356809107513606},
                          {0.15087462705261923, 0.12432607636646453},
                          {0.09086197752443204, 0.007373645295331757},
                          {0.12318156265055501, 0.10170602101718566},
                          {0.12124532321617587, 0.03953517502409465},
                          {0.138876769212277, 0.11854935269595822},
                          {0.12925271382763254, 0.060781917914583136},
                          {0.15742743525849087, 0.10319794072616913},
                          {0.01864314534132693, 0.0027897754617813627},
                          {0.11749825907490032, 0.0032929785605602564},
                          {0.05904709958052064, 0.14317867937370143},
                          {0.04276657403541619, 0.06090880292535508},
                          {0.10327586599056611, 0.11172312107823236},
                          {0.05723058479760559, -0.007087842744834096},
                          {0.12176051180844974, 0.021969787377981238},
                          {0.09729792592974744, 0.048917124931394426},
                          {0.18106921913675206, 0.11943187731292407},
                          {0.14388384105689692, 0.014642723623722667}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyDoubleExcitationMinus({0, 1, 2, 3}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("DoubleExcitationMinus", {0, 1, 2, 3}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }
}

// DoubleExcitationPlus Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyDoubleExcitationPlus",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 6;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.14836184869237898, 0.06928854101488939},
                          {0.07094825327558218, 0.061139592727065234},
                          {0.10679034746239055, 0.12478353692681407},
                          {0.11571545213303741, 0.06594226247249863},
                          {0.1190627426132508, 0.07484925755324358},
                          {0.00906902232517842, 0.1237218622207207},
                          {0.1305100520050695, 0.11711448904556603},
                          {0.05088581583296845, 0.08008722296156004},
                          {0.07316105587189087, 0.06352039635890594},
                          {0.11095039687205051, 0.03583892505512524},
                          {0.03570968538901324, 0.09638794194868609},
                          {0.0713600372820485, 0.056471298297101645},
                          {0.14628197548279212, 0.005040172811679685},
                          {0.070043404380343, 0.07117993551880243},
                          {0.06074339783032044, 0.12942393723357834},
                          {0.13268568980573536, 0.08036089556546508},
                          {0.036258177794672615, 0.08363767253314282},
                          {0.13825718318791222, 0.07482394855854033},
                          {0.07835567694876108, 0.022806916071867203},
                          {0.12820641894084636, 0.1050400222703431},
                          {0.07756072170440836, 0.020371250936538677},
                          {0.08014481803707861, 0.06974261931716805},
                          {0.10823246498880147, 0.11082881021426993},
                          {0.0233312644662286, 0.028272496102782756},
                          {0.008840824821587321, 0.1265209337189973},
                          {0.02617377406378093, 0.05258049528203285},
                          {0.0508392238465671, 0.01130963009263731},
                          {0.11457119366932077, 0.03413193019742667},
                          {0.12112734694895362, 0.0186002687068472},
                          {0.09252244815846217, 0.02966503116247933},
                          {0.1437401906964705, 0.10570310943862452},
                          {0.03392060130372184, 0.04250381993713222},
                          {0.08626107014172368, 0.06021136873328912},
                          {0.01166779452451774, 0.03645181951160576},
                          {0.1119077793976233, 0.10963382996954998},
                          {0.13350973802938118, 0.005648970975886418},
                          {0.05996197257704452, 0.1297269037363724},
                          {0.05212466979002961, 0.08441484000268347},
                          {0.015560963020448076, 0.09045774083745448},
                          {0.0736093207875478, 0.04994035230603118},
                          {0.040527605427232036, 0.12006575976853436},
                          {0.0026016539235101596, 0.001501879850125351},
                          {0.08752150598418447, 0.0714082719633339},
                          {0.0649944068764156, 0.04190325980719882},
                          {0.09601550206852992, 0.13632483092843833},
                          {0.0503907955404767, 0.08748233712665994},
                          {0.06295764772507972, 0.14904685877449778},
                          {0.12316740102643296, 0.15182199339140806},
                          {0.06305718075195604, 0.00650192311401943},
                          {0.11148843236383083, 0.08934571675171843},
                          {0.11139801671381279, 0.014103743906341755},
                          {0.114804641395206, 0.10467054960473593},
                          {0.11460076196121058, 0.08524887644475407},
                          {0.13378710190861637, 0.1324168558193471},
                          {0.017717280830810585, 0.006437986898650014},
                          {0.11450190282689218, 0.02657063872874804},
                          {0.029424876402037975, 0.1520554860726633},
                          {0.02981337874417589, 0.06819108868747294},
                          {0.07902126685601678, 0.13001384412451206},
                          {0.057497920365339114, 0.00442340419954388},
                          {0.11496868516821211, 0.04572193372968664},
                          {0.08564011282247964, 0.06727215306155458},
                          {0.15373243881038598, 0.15302409187671534},
                          {0.1381066836079441, 0.042936150447173645}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.13163898125494572, 0.09738233247130822},
                          {0.057387449819444845, 0.07401611341396051},
                          {0.07987098860845128, 0.14351214087251407},
                          {0.10030814201885441, 0.08761691895444915},
                          {0.1018191627927512, 0.09701137110176429},
                          {-0.01569149389750591, 0.123057398707116},
                          {0.10464148287740278, 0.14070834118838926},
                          {0.033960612393069065, 0.08860026151524122},
                          {0.059083151023155075, 0.07678907549292086},
                          {0.10161868051540206, 0.05716697373030205},
                          {0.015848541231338924, 0.10156101970994648},
                          {0.05871847249182722, 0.06952268292062753},
                          {0.13083854720946214, 0.003647972205328197},
                          {0.04649786738228028, 0.05201082205840395},
                          {0.03740120460558103, 0.12404209389167492},
                          {0.107232648658259, 0.05796419986517863},
                          {0.018919187798568132, 0.08917387541647964},
                          {0.12063602060137749, 0.10079991326589043},
                          {0.07226274540810915, 0.03791916608907167},
                          {0.10478259533748843, 0.12841689861747219},
                          {0.07196752830471602, 0.035374118868715436},
                          {0.06469153803767216, 0.08427472760997973},
                          {0.0840567360148148, 0.13012208414254106},
                          {0.0172493146414798, 0.032344135200592365},
                          {-0.0164712323053325, 0.12575533928623736},
                          {0.015205909365325284, 0.056732312249891975},
                          {0.04757894749336592, 0.021184385041279178},
                          {0.10550642996761084, 0.05621334639838341},
                          {0.11501756147065208, 0.04229379065777709},
                          {0.0847846272510933, 0.04745507843212733},
                          {0.11987499077900868, 0.13215285222871326},
                          {0.024800242174911988, 0.04839555651214201},
                          {0.07257943948222521, 0.07614857918029388},
                          {0.004193356861965808, 0.03804324293486627},
                          {0.08789619475549125, 0.12968109619757257},
                          {0.12972615477547622, 0.0320606579615826},
                          {0.03299396811277396, 0.13905360756387994},
                          {0.034315006969713784, 0.09308773662554466},
                          {-0.0027203990620455042, 0.09174609461161305},
                          {0.06222041874916132, 0.06356878468189904},
                          {0.015866367414552128, 0.1257240305412106},
                          {0.0022514165927828515, 0.0019888110889851782},
                          {0.07159026925324415, 0.0873726997567152},
                          {0.055373953342162115, 0.05398037975919321},
                          {0.06701802160071006, 0.15268274606472707},
                          {0.032006277166711235, 0.09574962039753666},
                          {0.03209164666501784, 0.15858359855908166},
                          {0.09054987939853273, 0.1732652466418413},
                          {0.09086197752443204, 0.007373645295331757},
                          {0.12318156265055501, 0.10170602101718566},
                          {0.12124532321617587, 0.03953517502409465},
                          {0.138876769212277, 0.11854935269595822},
                          {0.09538003935901224, 0.10631723128945876},
                          {0.10481309899525929, 0.15635672873656006},
                          {0.01608508424368259, 0.009829536114111973},
                          {0.1069407170448249, 0.04878901137931985},
                          {-0.0013704237430269064, 0.154870320380742},
                          {0.01567161812396236, 0.07275481093304534},
                          {0.05161633917794568, 0.14312132548795797},
                          {0.05547299529328669, 0.015758303977345192},
                          {0.10359341985498377, 0.0676512908674948},
                          {0.0705680986699835, 0.08294525273870558},
                          {0.12026683127941407, 0.1805157187928282},
                          {0.1268236485043848, 0.06951784844515367}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyDoubleExcitationPlus({0, 1, 2, 3}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("DoubleExcitationPlus", {0, 1, 2, 3}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }
}
// MultiRZ Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyMultiRZ",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();

    const std::vector<PrecisionT> angles{{0.4}};

    std::vector<std::vector<cp_t>> initstate{
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.09791401219094054, 0.114929230389338},
                          {0.2846159036261283, 0.20896501819533683},
                          {0.21084550974310806, 0.1960807418253313},
                          {0.20984254850784573, 0.09607341335335315},
                          {0.1527039474967227, 0.04968393919295135},
                          {-0.023374395680686586, 0.1874609940644216},
                          {0.19628754532973963, 0.06090861117447334},
                          {0.14429060004046249, 0.16020271083802284},
                          {-0.029305014762791147, 0.25299877929913567},
                          {0.2259205020010718, 0.35422096098183514},
                          {0.16466399912430643, 0.047542289852867514},
                          {0.08164095607238234, 0.21137551233950838},
                          {0.24238671886852403, 0.19355813861638085},
                          {0.14693482451317494, 0.02855139473814395},
                          {0.23776378523742325, 0.18593363133959642}},
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.045429227014373304, 0.1439863434985753},
                          {0.18077379611678607, 0.3033041807555934},
                          {0.21084550974310806, 0.1960807418253313},
                          {0.23069053568073156, 0.0067729362147416275},
                          {0.1527039474967227, 0.04968393919295135},
                          {-0.023374395680686586, 0.1874609940644216},
                          {0.2045117320076819, -0.02033742456644565},
                          {0.19528631758643603, 0.09136706180794868},
                          {-0.029305014762791147, 0.25299877929913567},
                          {0.2259205020010718, 0.35422096098183514},
                          {0.1701794264139849, -0.020333832827845056},
                          {0.08164095607238234, 0.21137551233950838},
                          {0.1478778627338016, 0.2726686858107655},
                          {0.12421749871021645, 0.08351669180701665},
                          {0.23776378523742325, 0.18593363133959642}},
        std::vector<cp_t>{{0.17869909972402495, 0.049084004040150196},
                          {0.09791401219094054, 0.114929230389338},
                          {0.18077379611678607, 0.3033041807555934},
                          {0.11784413734476112, 0.2627094318578463},
                          {0.20984254850784573, 0.09607341335335315},
                          {0.1527039474967227, 0.04968393919295135},
                          {0.0514715054362289, 0.18176542794818454},
                          {0.2045117320076819, -0.02033742456644565},
                          {0.19528631758643603, 0.09136706180794868},
                          {0.0715306592140959, 0.24443921741303512},
                          {0.2259205020010718, 0.35422096098183514},
                          {0.16466399912430643, 0.047542289852867514},
                          {-0.0071172014685187066, 0.22648222528149717},
                          {0.1478778627338016, 0.2726686858107655},
                          {0.14693482451317494, 0.02855139473814395},
                          {0.23776378523742325, 0.18593363133959642}}};

    std::vector<cp_t> localstate(subSvLength, {0.0, 0.0});

    auto init_localstate = mpi_manager.scatter<cp_t>(initstate[0], 0);
    auto expected_localstate0 =
        mpi_manager.scatter<cp_t>(expected_results[0], 0);
    auto expected_localstate1 =
        mpi_manager.scatter<cp_t>(expected_results[1], 0);
    auto expected_localstate2 =
        mpi_manager.scatter<cp_t>(expected_results[2], 0);

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyMultiRZ({0, 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyMultiRZ({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyMultiRZ({1, num_qubits - 2}, false, angles[0]);
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("MultiRZ", {0, 1}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("MultiRZ", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        sv.CopyHostDataToGpu(init_localstate, false);
        sv.applyOperation("MultiRZ", {1, num_qubits - 2}, false, {angles[0]});
        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}
*/