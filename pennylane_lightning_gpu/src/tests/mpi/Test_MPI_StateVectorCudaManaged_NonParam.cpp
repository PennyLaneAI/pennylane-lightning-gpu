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
#define msb_2qubit                                                             \
    { num_qubits - 2, num_qubits - 1 }
#define mlsb_2qubit                                                            \
    { 0, num_qubits - 1 }
#define lsb_3qbit                                                              \
    { 0, 1, 2 }
#define msb_3qubit                                                             \
    { num_qubits - 3, num_qubits - 2, num_qubits - 1 }
#define mlsb_3qubit                                                            \
    { 0, num_qubits - 2, num_qubits - 1 }

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SetStateVector",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_state(Pennylane::Util::exp2(num_qubits));
    std::vector<cp_t> expected_state(Pennylane::Util::exp2(num_qubits));
    std::vector<cp_t> local_state(subSvLength);

    using index_type =
        typename std::conditional<std::is_same<PrecisionT, float>::value,
                                  int32_t, int64_t>::type;

    std::vector<index_type> indices(Pennylane::Util::exp2(num_qubits));

    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        init_state = createRandomState<PrecisionT>(re, num_qubits);
        expected_state = init_state;
        for (size_t i = 0; i < Pennylane::Util::exp2(num_qubits - 1); i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
            indices[i * 2] = i * 2 + 1;
            indices[i * 2 + 1] = i * 2;
        }
    }
    mpi_manager.Barrier();

    auto expected_local_state = mpi_manager.scatter<cp_t>(expected_state, 0);
    mpi_manager.Bcast<index_type>(indices, 0);
    mpi_manager.Bcast<cp_t>(init_state, 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        // The setStates will shuffle the state vector values on the device with
        // the following indices and values setting on host. For example, the
        // values[i] is used to set the indices[i] th element of state vector on
        // the device. For example, values[2] (init_state[5]) will be copied to
        // indices[2]th or (4th) element of the state vector.

        sv.template setStateVector<index_type>(
            init_state.size(), init_state.data(), indices.data(), false);

        mpi_manager.Barrier();
        sv.CopyGpuDataToHost(local_state.data(),
                             static_cast<std::size_t>(subSvLength));
        mpi_manager.Barrier();

        CHECK(expected_local_state == Pennylane::approx(local_state));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SetIthStates",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    int index;
    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        std::uniform_int_distribution<> distr(
            0, Pennylane::Util::exp2(num_qubits) - 1);
        index = distr(re);
    }
    mpi_manager.Bcast(index, 0);

    std::vector<cp_t> expected_state(Pennylane::Util::exp2(num_qubits), {0, 0});
    if (mpi_manager.getRank() == 0) {
        expected_state[index] = {1.0, 0};
    }

    auto expected_local_state = mpi_manager.scatter(expected_state, 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nLocalIndexBits,
                                          nLocalIndexBits);
        std::complex<PrecisionT> values = {1.0, 0};
        sv.setBasisState(values, index, false);

        std::vector<cp_t> h_sv0(subSvLength, {0.0, 0.0});
        sv.CopyGpuDataToHost(h_sv0.data(),
                             static_cast<std::size_t>(subSvLength));

        CHECK(expected_local_state == Pennylane::approx(h_sv0));
    }
}

#define PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, NUM_QUBITS, GATE_METHOD,    \
                                         GATE_NAME, WIRE)                      \
    {                                                                          \
        using cp_t = std::complex<TestType>;                                   \
        using PrecisionT = TestType;                                           \
        MPIManager mpi_manager(MPI_COMM_WORLD);                                \
        int nGlobalIndexBits =                                                 \
            std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - \
            1;                                                                 \
        int nLocalIndexBits = (NUM_QUBITS)-nGlobalIndexBits;                   \
        int subSvLength = 1 << nLocalIndexBits;                                \
        int svLength = 1 << (NUM_QUBITS);                                      \
        mpi_manager.Barrier();                                                 \
        std::vector<cp_t> init_sv(svLength);                                   \
        std::vector<cp_t> expected_sv(svLength);                               \
        if (mpi_manager.getRank() == 0) {                                      \
            std::mt19937 re{1337};                                             \
            auto random_sv = createRandomState<PrecisionT>(re, (NUM_QUBITS));  \
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
            SECTION("Operation on target wire") {                              \
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits, \
                                                nLocalIndexBits);              \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.GATE_METHOD(WIRE, false);                                   \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                                                                               \
                SVDataGPU<TestType> svdat{(NUM_QUBITS), init_sv};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.cuda_sv.GATE_METHOD(WIRE, false);                    \
                    svdat.cuda_sv.CopyGpuDataToHost(                           \
                        expected_sv.data(),                                    \
                        static_cast<std::size_t>(svLength));                   \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state == Pennylane::approx(expected_local_sv));    \
            }                                                                  \
        }                                                                      \
        SECTION("Apply using dispatcher") {                                    \
            SECTION("Operation on target wire") {                              \
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits, \
                                                nLocalIndexBits);              \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.applyOperation(GATE_NAME, WIRE, false);                     \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                SVDataGPU<TestType> svdat{(NUM_QUBITS), init_sv};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.cuda_sv.applyOperation(GATE_NAME, WIRE, false);      \
                    svdat.cuda_sv.CopyGpuDataToHost(                           \
                        expected_sv.data(),                                    \
                        static_cast<std::size_t>(svLength));                   \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state == Pennylane::approx(expected_local_sv));    \
            }                                                                  \
        }                                                                      \
    }

TEMPLATE_TEST_CASE("StateVectorCudaMPI::GateOpsNonParam",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliX,
                                     "PauliX", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliX,
                                     "PauliX", {num_qubits - 1});

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliY,
                                     "PauliY", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliY,
                                     "PauliY", {num_qubits - 1});

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliZ,
                                     "PauliZ", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliZ,
                                     "PauliZ", {num_qubits - 1});

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyS, "S",
                                     lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyS, "S",
                                     {num_qubits - 1});

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyT, "T",
                                     lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyT, "T",
                                     {num_qubits - 1});
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     msb_2qubit);

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     msb_2qubit);

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CY",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CY",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CY",
                                     msb_2qubit);

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CZ",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CZ",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CZ",
                                     msb_2qubit);

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", lsb_3qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", mlsb_3qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", msb_3qubit);

    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     lsb_3qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     mlsb_3qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     msb_3qubit);
}
/*
TEMPLATE_TEST_CASE("StateVectorCudaMPI::GateOps",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
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

    if(mpi_manager.getRank() == 0){
        std::mt19937 re{1337};
        auto random_sv = createRandomState<PrecisionT>(re, num_qubits);
        init_sv=random_sv;
    }

    auto local_state = mpi_manager.scatter(init_sv, 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Gate: PauliX"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyPauliX({0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyPauliX({0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyPauliX({num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyPauliX({num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("PauliX", {0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("PauliX", {0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("PauliX", {num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("PauliX", {num_qubits - 1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: PauliY"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyPauliY({0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyPauliY({0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyPauliY({num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyPauliY({num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("PauliY", {0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("PauliY", {0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("PauliY", {num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("PauliY", {num_qubits - 1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: PauliZ"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyPauliZ({0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyPauliZ({0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyPauliZ({num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyPauliZ({num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("PauliZ", {0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("PauliZ", {0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("PauliZ", {num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("PauliZ", {num_qubits - 1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

        SECTION("Gate: Hadamard"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyHadamard({0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyHadamard({0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyHadamard({num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyHadamard({num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("Hadamard", {0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("Hadamard", {0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("Hadamard", {num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("Hadamard", {num_qubits - 1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: T"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyT({0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyT({0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyT({num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyT({num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("T", {0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("T", {0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("T", {num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("T", {num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: S"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyS({0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyS({0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyS({num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyS({num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("S", {0}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("S", {0}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("S", {num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("S", {num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: CNOT"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCNOT({0, 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCNOT({0,1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCNOT({0, num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCNOT({0,num_qubits-1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCNOT({num_qubits - 2,num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCNOT({num_qubits - 2,num_qubits - 1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CNOT", {0,1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CNOT", {0,1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CNOT", {0,num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CNOT", {0,num_qubits-1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CNOT", {num_qubits - 2,num_qubits - 1},
false); sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CNOT", {num_qubits -
2,num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: SWAP"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySWAP({0, 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applySWAP({0,1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySWAP({0, num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applySWAP({0,num_qubits-1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applySWAP({num_qubits - 2,num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applySWAP({num_qubits - 2,num_qubits - 1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SWAP", {0,1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("SWAP", {0,1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SWAP", {0,num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("SWAP", {0,num_qubits-1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("SWAP", {num_qubits - 2,num_qubits - 1},
false); sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("SWAP", {num_qubits -
2,num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }
    SECTION("Gate: CY"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCY({0, 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCY({0,1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCY({0, num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCY({0,num_qubits-1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCY({num_qubits - 2,num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCY({num_qubits - 2,num_qubits - 1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CY", {0,1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CY", {0,1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CY", {0,num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CY", {0,num_qubits-1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CY", {num_qubits - 2,num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CY", {num_qubits -
2,num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: CZ"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCZ({0, 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCZ({0,1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCZ({0, num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCZ({0,num_qubits-1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCZ({num_qubits - 2,num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCZ({num_qubits - 2,num_qubits - 1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CZ", {0,1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CZ", {0,1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CZ", {0,num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CZ", {0,num_qubits-1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CZ", {num_qubits - 2,num_qubits - 1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CZ", {num_qubits -
2,num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: Toffoli"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyToffoli({0, 1,2}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyToffoli({0,1,2}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyToffoli({0, num_qubits - 2,num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyToffoli({0,num_qubits - 2,num_qubits-1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyToffoli({num_qubits - 3,num_qubits - 2,num_qubits - 1},
false); sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyToffoli({num_qubits - 3,num_qubits -
2,num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("Toffoli", {0,1,2}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("Toffoli", {0,1,2}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("Toffoli", {0,num_qubits - 2,num_qubits-1},
false); sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("Toffoli", {0,num_qubits - 2,
num_qubits-1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("Toffoli", {num_qubits - 3,num_qubits -
2,num_qubits - 1}, false); sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("Toffoli", {num_qubits -
3,num_qubits - 2,num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }

    SECTION("Gate: CSWAP"){
        SECTION("Apply directly") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCSWAP({0, 1,2}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCSWAP({0,1,2}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCSWAP({0, num_qubits - 2,num_qubits-1}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCSWAP({0,num_qubits - 2,num_qubits-1},
false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyCSWAP({num_qubits - 3,num_qubits - 2,num_qubits - 1},
false); sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyCSWAP({num_qubits - 3,num_qubits -
2,num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }

        SECTION("Apply using dispatcher") {
            SECTION("Operation on globalQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CSWAP", {0,1,2}, false);
                sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CSWAP", {0,1,2}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on global and local Qubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CSWAP", {0,num_qubits - 2,num_qubits-1},
false); sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));

                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CSWAP", {0,num_qubits -
2,num_qubits-1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }

            SECTION("Operation on localQubits") {
                StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
                sv.CopyHostDataToGpu(local_state, false);
                sv.applyOperation("CSWAP", {num_qubits - 3,num_qubits -
2,num_qubits - 1}, false); sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
                SVDataGPU<TestType> svdat{num_qubits, init_sv};
                if(mpi_manager.getRank() == 0){
                    svdat.cuda_sv.applyOperation("CSWAP", {num_qubits -
3,num_qubits - 2,num_qubits - 1}, false);
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),static_cast<std::size_t>(svLength));
                }
                auto expected_local_sv = mpi_manager.scatter(expected_sv,0);
                CHECK(local_state == Pennylane::approx(expected_local_sv));
            }
        }
    }
}

*/

/*
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyPauliX",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {
        {0.0, 0.0}, {0.1, 0.0}, {0.2, 0.0}, {0.3, 0.0}, {0.4, 0.0}, {0.5, 0.0},
        {0.6, 0.0}, {0.7, 0.0}, {0.8, 0.0}, {0.9, 0.0}, {1.0, 0.0}, {1.1, 0.0},
        {1.2, 0.0}, {1.3, 0.0}, {1.4, 0.0}, {1.5, 0.0}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.8, 0.0},
                          {0.9, 0.0},
                          {1.0, 0.0},
                          {1.1, 0.0},
                          {1.2, 0.0},
                          {1.3, 0.0},
                          {1.4, 0.0},
                          {1.5, 0.0},
                          {0.0, 0.0},
                          {0.1, 0.0},
                          {0.2, 0.0},
                          {0.3, 0.0},
                          {0.4, 0.0},
                          {0.5, 0.0},
                          {0.6, 0.0},
                          {0.7, 0.0}},
        std::vector<cp_t>{{0.1, 0.0},
                          {0.0, 0.0},
                          {0.3, 0.0},
                          {0.2, 0.0},
                          {0.5, 0.0},
                          {0.4, 0.0},
                          {0.7, 0.0},
                          {0.6, 0.0},
                          {0.9, 0.0},
                          {0.8, 0.0},
                          {1.1, 0.0},
                          {1.0, 0.0},
                          {1.3, 0.0},
                          {1.2, 0.0},
                          {1.5, 0.0},
                          {1.4, 0.0}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyPauliX({0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyPauliX({num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("PauliX", {0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("PauliX", {num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyPauliY",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.12834320562185453, -0.17324175995006008},
                          {0.2537776554975786, -0.021542232643170886},
                          {0.30227665008366594, -0.2917899745322105},
                          {0.013880922806771745, -0.17082687702494623},
                          {0.2233816291263903, -0.03801974084659355},
                          {0.2378546697582974, -0.1991010562067874},
                          {0.0571737109901294, -0.13833362414043807},
                          {0.22946370987301284, -0.1960850292216881},
                          {-0.08360762242222763, 0.1653855288944372},
                          {-0.13209080879903976, 0.0731293375604395},
                          {-0.2613440813782711, 0.23742759434160687},
                          {-0.2340607179431313, 0.16768740742688235},
                          {-0.052469062762363974, 0.2247465091396771},
                          {-0.018355977199570113, 0.1595307101966878},
                          {-0.18836803047905595, 0.01433428625707798},
                          {-0.02069817884076428, 0.20447553584586473}},
        std::vector<cp_t>{{0.13209080879903976, -0.0731293375604395},
                          {-0.08360762242222763, 0.1653855288944372},
                          {0.2340607179431313, -0.16768740742688235},
                          {-0.2613440813782711, 0.23742759434160687},
                          {0.018355977199570113, -0.1595307101966878},
                          {-0.052469062762363974, 0.2247465091396771},
                          {0.02069817884076428, -0.20447553584586473},
                          {-0.18836803047905595, 0.01433428625707798},
                          {0.2537776554975786, -0.021542232643170886},
                          {-0.12834320562185453, 0.17324175995006008},
                          {0.013880922806771745, -0.17082687702494623},
                          {-0.30227665008366594, 0.2917899745322105},
                          {0.2378546697582974, -0.1991010562067874},
                          {-0.2233816291263903, 0.03801974084659355},
                          {0.22946370987301284, -0.1960850292216881},
                          {-0.0571737109901294, 0.13833362414043807}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyPauliY({0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyPauliY({num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("PauliY", {0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("PauliY", {num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyPauliZ",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {-0.17324175995006008, -0.12834320562185453},
                          {-0.021542232643170886, -0.2537776554975786},
                          {-0.2917899745322105, -0.30227665008366594},
                          {-0.17082687702494623, -0.013880922806771745},
                          {-0.03801974084659355, -0.2233816291263903},
                          {-0.1991010562067874, -0.2378546697582974},
                          {-0.13833362414043807, -0.0571737109901294},
                          {-0.1960850292216881, -0.22946370987301284}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {-0.0731293375604395, -0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {-0.16768740742688235, -0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {-0.1595307101966878, -0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {-0.20447553584586473, -0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {-0.021542232643170886, -0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {-0.17082687702494623, -0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {-0.1991010562067874, -0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {-0.1960850292216881, -0.22946370987301284}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyPauliZ({0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyPauliZ({num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("PauliZ", {0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("PauliZ", {num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyHadamard",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.23944565223675973, 0.14987186778807435},
                          {0.0669429092765512, 0.272850207750178},
                          {0.374213331673735, 0.39854004123405773},
                          {0.23936574606439984, 0.17532121551277524},
                          {0.1858037972322515, 0.1950558948295391},
                          {0.2535909539728071, 0.1811682858761006},
                          {0.10795251471166382, 0.17362423045721506},
                          {0.2832390918351819, 0.17689116790394624},
                          {-0.005555194253999293, -0.03163283424079116},
                          {0.03647759170878253, -0.08604559448174463},
                          {-0.03844000767421707, -0.02894369690296894},
                          {-0.0022199402421199665, 0.15569062622118462},
                          {0.13203576408908857, -0.12085343466599885},
                          {-0.02798045999763947, -0.15520901396985723},
                          {-0.08768077267996587, 0.0927683929637744},
                          {0.005932984131545943, -0.14761952267091358}},
        std::vector<cp_t>{{0.16865547948404708, 0.15252182340785828},
                          {0.06523497849871336, -0.03428278986057509},
                          {0.2864595649108989, 0.3503040930325243},
                          {0.04931375908861903, 0.01929225129856446},
                          {0.27172502764825385, 0.05008086603489181},
                          {0.0461145336730862, 0.024121594128648453},
                          {0.1547219089992129, 0.14783213432701106},
                          {-0.13445016696751494, 0.1185604890939784},
                          {0.13773308202926385, 0.27020025213039406},
                          {0.10726776446149519, -0.08869555010152856},
                          {0.32711951282723595, 0.22355716371430864},
                          {0.08553382652071613, 0.20392657442271803},
                          {0.16766972355680476, 0.32614331467074786},
                          {-0.11390169041364184, -0.010233985175209931},
                          {0.2364696975476328, 0.20268326403415024},
                          {-0.04083641015600313, -0.12182742654070958}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyHadamard({0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyHadamard({num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("Hadamard", {0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("Hadamard", {num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyS",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {-0.12834320562185453, 0.17324175995006008},
                          {-0.2537776554975786, 0.021542232643170886},
                          {-0.30227665008366594, 0.2917899745322105},
                          {-0.013880922806771745, 0.17082687702494623},
                          {-0.2233816291263903, 0.03801974084659355},
                          {-0.2378546697582974, 0.1991010562067874},
                          {-0.0571737109901294, 0.13833362414043807},
                          {-0.22946370987301284, 0.1960850292216881}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {-0.13209080879903976, 0.0731293375604395},
                          {0.23742759434160687, 0.2613440813782711},
                          {-0.2340607179431313, 0.16768740742688235},
                          {0.2247465091396771, 0.052469062762363974},
                          {-0.018355977199570113, 0.1595307101966878},
                          {0.01433428625707798, 0.18836803047905595},
                          {-0.02069817884076428, 0.20447553584586473},
                          {0.17324175995006008, 0.12834320562185453},
                          {-0.2537776554975786, 0.021542232643170886},
                          {0.2917899745322105, 0.30227665008366594},
                          {-0.013880922806771745, 0.17082687702494623},
                          {0.03801974084659355, 0.2233816291263903},
                          {-0.2378546697582974, 0.1991010562067874},
                          {0.13833362414043807, 0.0571737109901294},
                          {-0.22946370987301284, 0.1960850292216881}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyS({0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyS({num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("S", {0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("S", {num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyT",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.03174807223094678, 0.2132527742598123},
                          {-0.164215242332077, 0.19468055989984567},
                          {-0.007415199394537275, 0.4200685387424894},
                          {0.11097754850746462, 0.1306081377990552},
                          {-0.13107064817618752, 0.18483868131935047},
                          {-0.02740294293775558, 0.30897435690820224},
                          {0.05738872494909453, 0.1382445624425352},
                          {-0.023602291435611905, 0.30090839913924794}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {-0.04169205614154981, 0.14511255712688356},
                          {0.23742759434160687, 0.2613440813782711},
                          {-0.046933017955839984, 0.2840788237781199},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.09982561103446216, 0.1257848829407055},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.12995021536684762, 0.15922186059988025},
                          {0.17324175995006008, 0.12834320562185453},
                          {-0.164215242332077, 0.19468055989984567},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.11097754850746462, 0.1306081377990552},
                          {0.03801974084659355, 0.2233816291263903},
                          {-0.02740294293775558, 0.30897435690820224},
                          {0.13833362414043807, 0.0571737109901294},
                          {-0.023602291435611905, 0.30090839913924794}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyT({0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyT({num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("T", {0}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("T", {num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyCNOT",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.1960850292216881, 0.22946370987301284},
                          {0.13833362414043807, 0.0571737109901294}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    auto expected_sv2 = mpi_manager.scatter(expected_sv[2], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCNOT({0, 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCNOT({num_qubits - 2, num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }

        SECTION("Operation on both global and local qubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCNOT({1, num_qubits - 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv2));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CNOT", {0, 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CNOT", {num_qubits - 2, num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CNOT", {1, num_qubits - 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv2));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applySWAP",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.1960850292216881, 0.22946370987301284}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.13833362414043807, 0.0571737109901294},
                          {0.1960850292216881, 0.22946370987301284}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    auto expected_sv2 = mpi_manager.scatter(expected_sv[2], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applySWAP({0, 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applySWAP({num_qubits - 2, num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }

        SECTION("Operation on both global and local qubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applySWAP({1, num_qubits - 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv2));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("SWAP", {0, 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("SWAP", {num_qubits - 2, num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("SWAP", {1, num_qubits - 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv2));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyCY",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {0.20447553584586473, 0.02069817884076428},
                          {0.2233816291263903, -0.03801974084659355},
                          {0.2378546697582974, -0.1991010562067874},
                          {0.0571737109901294, -0.13833362414043807},
                          {0.22946370987301284, -0.1960850292216881},
                          {-0.12834320562185453, 0.17324175995006008},
                          {-0.2537776554975786, 0.021542232643170886},
                          {-0.30227665008366594, 0.2917899745322105},
                          {-0.013880922806771745, 0.17082687702494623}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.2340607179431313, -0.16768740742688235},
                          {-0.2613440813782711, 0.23742759434160687},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.02069817884076428, -0.20447553584586473},
                          {-0.18836803047905595, 0.01433428625707798},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.013880922806771745, -0.17082687702494623},
                          {-0.30227665008366594, 0.2917899745322105},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.22946370987301284, -0.1960850292216881},
                          {-0.0571737109901294, 0.13833362414043807}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.18836803047905595, -0.01433428625707798},
                          {0.02069817884076428, -0.20447553584586473},
                          {-0.052469062762363974, 0.2247465091396771},
                          {-0.018355977199570113, 0.1595307101966878},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.0571737109901294, -0.13833362414043807},
                          {0.22946370987301284, -0.1960850292216881},
                          {-0.2233816291263903, 0.03801974084659355},
                          {-0.2378546697582974, 0.1991010562067874}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    auto expected_sv2 = mpi_manager.scatter(expected_sv[2], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCY({0, 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCY({num_qubits - 2, num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }

        SECTION("Operation on both global and local qubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCY({1, num_qubits - 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv2));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CY", {0, 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CY", {num_qubits - 2, num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CY", {1, num_qubits - 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv2));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyCZ",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.1653855288944372, 0.08360762242222763},
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
                                 {0.1960850292216881, 0.22946370987301284}};

    std::vector<std::vector<cp_t>> expected_sv = {
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
                          {-0.03801974084659355, -0.2233816291263903},
                          {-0.1991010562067874, -0.2378546697582974},
                          {-0.13833362414043807, -0.0571737109901294},
                          {-0.1960850292216881, -0.22946370987301284}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {-0.16768740742688235, -0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {0.01433428625707798, 0.18836803047905595},
                          {-0.20447553584586473, -0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {-0.17082687702494623, -0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {0.13833362414043807, 0.0571737109901294},
                          {-0.1960850292216881, -0.22946370987301284}},
        std::vector<cp_t>{{0.1653855288944372, 0.08360762242222763},
                          {0.0731293375604395, 0.13209080879903976},
                          {0.23742759434160687, 0.2613440813782711},
                          {0.16768740742688235, 0.2340607179431313},
                          {0.2247465091396771, 0.052469062762363974},
                          {0.1595307101966878, 0.018355977199570113},
                          {-0.01433428625707798, -0.18836803047905595},
                          {-0.20447553584586473, -0.02069817884076428},
                          {0.17324175995006008, 0.12834320562185453},
                          {0.021542232643170886, 0.2537776554975786},
                          {0.2917899745322105, 0.30227665008366594},
                          {0.17082687702494623, 0.013880922806771745},
                          {0.03801974084659355, 0.2233816291263903},
                          {0.1991010562067874, 0.2378546697582974},
                          {-0.13833362414043807, -0.0571737109901294},
                          {-0.1960850292216881, -0.22946370987301284}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    auto expected_sv2 = mpi_manager.scatter(expected_sv[2], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCZ({0, 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCZ({num_qubits - 2, num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }

        SECTION("Operation on both global and local qubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCZ({1, num_qubits - 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv2));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CZ", {0, 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CZ", {num_qubits - 2, num_qubits - 1}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }

        SECTION("Operation on localQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CZ", {1, num_qubits - 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv2));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyToffoli",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 6;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.14836184869237898, 0.06928854101488939},
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
                                 {0.1381066836079441, 0.042936150447173645}};

    std::vector<std::vector<cp_t>> expected_sv = {
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
                          {0.029424876402037975, 0.1520554860726633},
                          {0.02981337874417589, 0.06819108868747294},
                          {0.07902126685601678, 0.13001384412451206},
                          {0.057497920365339114, 0.00442340419954388},
                          {0.11496868516821211, 0.04572193372968664},
                          {0.08564011282247964, 0.06727215306155458},
                          {0.15373243881038598, 0.15302409187671534},
                          {0.1381066836079441, 0.042936150447173645},
                          {0.06305718075195604, 0.00650192311401943},
                          {0.11148843236383083, 0.08934571675171843},
                          {0.11139801671381279, 0.014103743906341755},
                          {0.114804641395206, 0.10467054960473593},
                          {0.11460076196121058, 0.08524887644475407},
                          {0.13378710190861637, 0.1324168558193471},
                          {0.017717280830810585, 0.006437986898650014},
                          {0.11450190282689218, 0.02657063872874804}},
        std::vector<cp_t>{{0.14836184869237898, 0.06928854101488939},
                          {0.07094825327558218, 0.061139592727065234},
                          {0.10679034746239055, 0.12478353692681407},
                          {0.11571545213303741, 0.06594226247249863},
                          {0.1190627426132508, 0.07484925755324358},
                          {0.00906902232517842, 0.1237218622207207},
                          {0.05088581583296845, 0.08008722296156004},
                          {0.1305100520050695, 0.11711448904556603},
                          {0.07316105587189087, 0.06352039635890594},
                          {0.11095039687205051, 0.03583892505512524},
                          {0.03570968538901324, 0.09638794194868609},
                          {0.0713600372820485, 0.056471298297101645},
                          {0.14628197548279212, 0.005040172811679685},
                          {0.070043404380343, 0.07117993551880243},
                          {0.13268568980573536, 0.08036089556546508},
                          {0.06074339783032044, 0.12942393723357834},
                          {0.036258177794672615, 0.08363767253314282},
                          {0.13825718318791222, 0.07482394855854033},
                          {0.07835567694876108, 0.022806916071867203},
                          {0.12820641894084636, 0.1050400222703431},
                          {0.07756072170440836, 0.020371250936538677},
                          {0.08014481803707861, 0.06974261931716805},
                          {0.0233312644662286, 0.028272496102782756},
                          {0.10823246498880147, 0.11082881021426993},
                          {0.008840824821587321, 0.1265209337189973},
                          {0.02617377406378093, 0.05258049528203285},
                          {0.0508392238465671, 0.01130963009263731},
                          {0.11457119366932077, 0.03413193019742667},
                          {0.12112734694895362, 0.0186002687068472},
                          {0.09252244815846217, 0.02966503116247933},
                          {0.03392060130372184, 0.04250381993713222},
                          {0.1437401906964705, 0.10570310943862452},
                          {0.08626107014172368, 0.06021136873328912},
                          {0.01166779452451774, 0.03645181951160576},
                          {0.1119077793976233, 0.10963382996954998},
                          {0.13350973802938118, 0.005648970975886418},
                          {0.05996197257704452, 0.1297269037363724},
                          {0.05212466979002961, 0.08441484000268347},
                          {0.0736093207875478, 0.04994035230603118},
                          {0.015560963020448076, 0.09045774083745448},
                          {0.040527605427232036, 0.12006575976853436},
                          {0.0026016539235101596, 0.001501879850125351},
                          {0.08752150598418447, 0.0714082719633339},
                          {0.0649944068764156, 0.04190325980719882},
                          {0.09601550206852992, 0.13632483092843833},
                          {0.0503907955404767, 0.08748233712665994},
                          {0.12316740102643296, 0.15182199339140806},
                          {0.06295764772507972, 0.14904685877449778},
                          {0.06305718075195604, 0.00650192311401943},
                          {0.11148843236383083, 0.08934571675171843},
                          {0.11139801671381279, 0.014103743906341755},
                          {0.114804641395206, 0.10467054960473593},
                          {0.11460076196121058, 0.08524887644475407},
                          {0.13378710190861637, 0.1324168558193471},
                          {0.11450190282689218, 0.02657063872874804},
                          {0.017717280830810585, 0.006437986898650014},
                          {0.029424876402037975, 0.1520554860726633},
                          {0.02981337874417589, 0.06819108868747294},
                          {0.07902126685601678, 0.13001384412451206},
                          {0.057497920365339114, 0.00442340419954388},
                          {0.11496868516821211, 0.04572193372968664},
                          {0.08564011282247964, 0.06727215306155458},
                          {0.1381066836079441, 0.042936150447173645},
                          {0.15373243881038598, 0.15302409187671534}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyToffoli({0, 1, 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on local and global Qubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyToffoli({num_qubits - 3, num_qubits - 2, num_qubits - 1},
                            false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("Toffoli", {0, 1, 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on local and global qubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("Toffoli",
                              {num_qubits - 3, num_qubits - 2, num_qubits - 1},
                              false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyCSWAP",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 6;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv = {{0.14836184869237898, 0.06928854101488939},
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
                                 {0.1381066836079441, 0.042936150447173645}};

    std::vector<std::vector<cp_t>> expected_sv = {
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
                          {0.06305718075195604, 0.00650192311401943},
                          {0.11148843236383083, 0.08934571675171843},
                          {0.11139801671381279, 0.014103743906341755},
                          {0.114804641395206, 0.10467054960473593},
                          {0.11460076196121058, 0.08524887644475407},
                          {0.13378710190861637, 0.1324168558193471},
                          {0.017717280830810585, 0.006437986898650014},
                          {0.11450190282689218, 0.02657063872874804},
                          {0.040527605427232036, 0.12006575976853436},
                          {0.0026016539235101596, 0.001501879850125351},
                          {0.08752150598418447, 0.0714082719633339},
                          {0.0649944068764156, 0.04190325980719882},
                          {0.09601550206852992, 0.13632483092843833},
                          {0.0503907955404767, 0.08748233712665994},
                          {0.06295764772507972, 0.14904685877449778},
                          {0.12316740102643296, 0.15182199339140806},
                          {0.029424876402037975, 0.1520554860726633},
                          {0.02981337874417589, 0.06819108868747294},
                          {0.07902126685601678, 0.13001384412451206},
                          {0.057497920365339114, 0.00442340419954388},
                          {0.11496868516821211, 0.04572193372968664},
                          {0.08564011282247964, 0.06727215306155458},
                          {0.15373243881038598, 0.15302409187671534},
                          {0.1381066836079441, 0.042936150447173645}},
        std::vector<cp_t>{{0.14836184869237898, 0.06928854101488939},
                          {0.07094825327558218, 0.061139592727065234},
                          {0.10679034746239055, 0.12478353692681407},
                          {0.11571545213303741, 0.06594226247249863},
                          {0.1190627426132508, 0.07484925755324358},
                          {0.1305100520050695, 0.11711448904556603},
                          {0.00906902232517842, 0.1237218622207207},
                          {0.05088581583296845, 0.08008722296156004},
                          {0.07316105587189087, 0.06352039635890594},
                          {0.11095039687205051, 0.03583892505512524},
                          {0.03570968538901324, 0.09638794194868609},
                          {0.0713600372820485, 0.056471298297101645},
                          {0.14628197548279212, 0.005040172811679685},
                          {0.06074339783032044, 0.12942393723357834},
                          {0.070043404380343, 0.07117993551880243},
                          {0.13268568980573536, 0.08036089556546508},
                          {0.036258177794672615, 0.08363767253314282},
                          {0.13825718318791222, 0.07482394855854033},
                          {0.07835567694876108, 0.022806916071867203},
                          {0.12820641894084636, 0.1050400222703431},
                          {0.07756072170440836, 0.020371250936538677},
                          {0.10823246498880147, 0.11082881021426993},
                          {0.08014481803707861, 0.06974261931716805},
                          {0.0233312644662286, 0.028272496102782756},
                          {0.008840824821587321, 0.1265209337189973},
                          {0.02617377406378093, 0.05258049528203285},
                          {0.0508392238465671, 0.01130963009263731},
                          {0.11457119366932077, 0.03413193019742667},
                          {0.12112734694895362, 0.0186002687068472},
                          {0.1437401906964705, 0.10570310943862452},
                          {0.09252244815846217, 0.02966503116247933},
                          {0.03392060130372184, 0.04250381993713222},
                          {0.08626107014172368, 0.06021136873328912},
                          {0.01166779452451774, 0.03645181951160576},
                          {0.1119077793976233, 0.10963382996954998},
                          {0.13350973802938118, 0.005648970975886418},
                          {0.05996197257704452, 0.1297269037363724},
                          {0.015560963020448076, 0.09045774083745448},
                          {0.05212466979002961, 0.08441484000268347},
                          {0.0736093207875478, 0.04994035230603118},
                          {0.040527605427232036, 0.12006575976853436},
                          {0.0026016539235101596, 0.001501879850125351},
                          {0.08752150598418447, 0.0714082719633339},
                          {0.0649944068764156, 0.04190325980719882},
                          {0.09601550206852992, 0.13632483092843833},
                          {0.06295764772507972, 0.14904685877449778},
                          {0.0503907955404767, 0.08748233712665994},
                          {0.12316740102643296, 0.15182199339140806},
                          {0.06305718075195604, 0.00650192311401943},
                          {0.11148843236383083, 0.08934571675171843},
                          {0.11139801671381279, 0.014103743906341755},
                          {0.114804641395206, 0.10467054960473593},
                          {0.11460076196121058, 0.08524887644475407},
                          {0.017717280830810585, 0.006437986898650014},
                          {0.13378710190861637, 0.1324168558193471},
                          {0.11450190282689218, 0.02657063872874804},
                          {0.029424876402037975, 0.1520554860726633},
                          {0.02981337874417589, 0.06819108868747294},
                          {0.07902126685601678, 0.13001384412451206},
                          {0.057497920365339114, 0.00442340419954388},
                          {0.11496868516821211, 0.04572193372968664},
                          {0.15373243881038598, 0.15302409187671534},
                          {0.08564011282247964, 0.06727215306155458},
                          {0.1381066836079441, 0.042936150447173645}}};

    auto local_state = mpi_manager.scatter(init_sv, 0);
    auto expected_sv0 = mpi_manager.scatter(expected_sv[0], 0);
    auto expected_sv1 = mpi_manager.scatter(expected_sv[1], 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    mpi_manager.Barrier();

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCSWAP({0, 1, 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on local and global Qubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyCSWAP({num_qubits - 3, num_qubits - 2, num_qubits - 1},
                          false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CSWAP", {0, 1, 2}, false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv0));
        }

        SECTION("Operation on local and global qubits") {
            StateVectorCudaMPI<TestType> sv(mpi_manager, nGlobalIndexBits,
                                            nLocalIndexBits);
            sv.CopyHostDataToGpu(local_state, false);
            sv.applyOperation("CSWAP",
                              {num_qubits - 3, num_qubits - 2, num_qubits - 1},
                              false);
            sv.CopyGpuDataToHost(local_state.data(),
                                 static_cast<std::size_t>(subSvLength));
            CHECK(local_state == Pennylane::approx(expected_sv1));
        }
    }
}
*/
