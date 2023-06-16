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

    size_t mpi_buffersize = num_qubits;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
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
    DevTag<int> dt_local(deviceId, 0);

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, dt_local, mpi_buffersize,
                                          nGlobalIndexBits, nLocalIndexBits);
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

    size_t mpi_buffersize = num_qubits;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
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
    DevTag<int> dt_local(deviceId, 0);

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, dt_local, mpi_buffersize,
                                          nLocalIndexBits, nLocalIndexBits);
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
        size_t mpi_buffersize = 8;                                             \
        size_t nGlobalIndexBits =                                              \
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;    \
        size_t nLocalIndexBits = (NUM_QUBITS)-nGlobalIndexBits;                \
        size_t subSvLength = 1 << nLocalIndexBits;                             \
        size_t svLength = 1 << (NUM_QUBITS);                                   \
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
        DevTag<int> dt_local(deviceId, 0);                                     \
        mpi_manager.Barrier();                                                 \
        SECTION("Apply directly") {                                            \
            SECTION("Operation on target wire") {                              \
                StateVectorCudaMPI<TestType> sv(                               \
                    mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,   \
                    nLocalIndexBits);                                          \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.GATE_METHOD(WIRE, false);                                   \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                                                                               \
                SVDataGPU<TestType> svdat{(NUM_QUBITS), init_sv};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.cuda_sv.GATE_METHOD(WIRE, false);                    \
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),        \
                                                    svLength);                 \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state == Pennylane::approx(expected_local_sv));    \
            }                                                                  \
        }                                                                      \
        SECTION("Apply using dispatcher") {                                    \
            SECTION("Operation on target wire") {                              \
                StateVectorCudaMPI<TestType> sv(                               \
                    mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,   \
                    nLocalIndexBits);                                          \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.applyOperation(GATE_NAME, WIRE, false);                     \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                SVDataGPU<TestType> svdat{(NUM_QUBITS), init_sv};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.cuda_sv.applyOperation(GATE_NAME, WIRE, false);      \
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),        \
                                                    svLength);                 \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state == Pennylane::approx(expected_local_sv));    \
            }                                                                  \
        }                                                                      \
    }

TEMPLATE_TEST_CASE("StateVectorCudaMPI::PauliX",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliX,
                                     "PauliX", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliX,
                                     "PauliX", {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::PauliY",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliY,
                                     "PauliY", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliY,
                                     "PauliY", {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::PauliZ",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliZ,
                                     "PauliZ", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliZ,
                                     "PauliZ", {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::S", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyS, "S",
                                     lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyS, "S",
                                     {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::T", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyT, "T",
                                     lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyT, "T",
                                     {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CNOT", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     msb_2qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SWAP", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     msb_2qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CY", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCY, "CY",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCY, "CY",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCY, "CY",
                                     msb_2qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CZ", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCZ, "CZ",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCZ, "CZ",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCZ, "CZ",
                                     msb_2qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::Toffoli",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", lsb_3qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", mlsb_3qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", msb_3qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CSWAP", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     lsb_3qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     mlsb_3qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     msb_3qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::expval_Identity",
                   "[StateVectorCudaMPI_Nonparam]", double) {
    using cp_t = std::complex<TestType>;
    using PrecisionT = TestType;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    size_t mpi_buffersize = num_qubits;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t svLength = 1 << num_qubits;
    mpi_manager.Barrier();
    std::vector<cp_t> init_sv(svLength);
    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        auto random_sv = createRandomState<PrecisionT>(re, num_qubits);
        init_sv = random_sv;
    }
    auto local_state = mpi_manager.scatter(init_sv, 0);
    mpi_manager.Barrier();
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                    nGlobalIndexBits, nLocalIndexBits);
    sv.CopyHostDataToGpu(local_state, false);
    PrecisionT mpi_result, result;
    std::vector<size_t> wire{0};

    std::vector<cp_t> matrix(4, cp_t{1.0, 0.0});

    matrix[1] = cp_t{0.0, 0.0};
    matrix[2] = cp_t{0.0, 0.0};

    mpi_result = sv.expval(wire, matrix).x;

    SVDataGPU<TestType> svdat{num_qubits, init_sv};
    if (mpi_manager.getRank() == 0) {
        result = svdat.cuda_sv.expval(wire, matrix).x;
    }
    mpi_manager.Bcast<PrecisionT>(result, 0);
    CHECK(result == Approx(mpi_result).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::expval_PauliX",
                   "[StateVectorCudaMPI_Nonparam]", double) {
    using cp_t = std::complex<TestType>;
    using PrecisionT = TestType;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    size_t mpi_buffersize = num_qubits;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t svLength = 1 << num_qubits;
    mpi_manager.Barrier();
    std::vector<cp_t> init_sv(svLength);
    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        auto random_sv = createRandomState<PrecisionT>(re, num_qubits);
        init_sv = random_sv;
    }
    auto local_state = mpi_manager.scatter(init_sv, 0);
    mpi_manager.Barrier();
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                    nGlobalIndexBits, nLocalIndexBits);
    sv.CopyHostDataToGpu(local_state, false);
    PrecisionT mpi_result, result;
    std::vector<size_t> wire{0};

    std::vector<cp_t> matrix(4, cp_t{0.0, 0.0});

    matrix[1] = cp_t{1.0, 0.0};
    matrix[2] = cp_t{1.0, 0.0};

    mpi_result = sv.expval(wire, matrix).x;

    SVDataGPU<TestType> svdat{num_qubits, init_sv};
    if (mpi_manager.getRank() == 0) {
        result = svdat.cuda_sv.expval(wire, matrix).x;
    }
    mpi_manager.Bcast<PrecisionT>(result, 0);
    CHECK(result == Approx(mpi_result).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::expval_PauliY",
                   "[StateVectorCudaMPI_Nonparam]", double) {
    using cp_t = std::complex<TestType>;
    using PrecisionT = TestType;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    size_t mpi_buffersize = num_qubits;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t svLength = 1 << num_qubits;
    mpi_manager.Barrier();
    std::vector<cp_t> init_sv(svLength);
    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        auto random_sv = createRandomState<PrecisionT>(re, num_qubits);
        init_sv = random_sv;
    }
    auto local_state = mpi_manager.scatter(init_sv, 0);
    mpi_manager.Barrier();
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                    nGlobalIndexBits, nLocalIndexBits);
    sv.CopyHostDataToGpu(local_state, false);
    PrecisionT mpi_result, result;
    std::vector<size_t> wire{0};

    std::vector<cp_t> matrix(4, cp_t{0.0, 0.0});

    matrix[1] = cp_t{0.0, -1.0};
    matrix[2] = cp_t{0.0, 1.0};

    mpi_result = sv.expval(wire, matrix).x;

    SVDataGPU<TestType> svdat{num_qubits, init_sv};
    if (mpi_manager.getRank() == 0) {
        result = svdat.cuda_sv.expval(wire, matrix).x;
    }
    mpi_manager.Bcast<PrecisionT>(result, 0);
    CHECK(result == Approx(mpi_result).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::expval_PauliZ",
                   "[StateVectorCudaMPI_Nonparam]", double) {
    using cp_t = std::complex<TestType>;
    using PrecisionT = TestType;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    size_t mpi_buffersize = num_qubits;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t svLength = 1 << num_qubits;
    mpi_manager.Barrier();
    std::vector<cp_t> init_sv(svLength);
    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        auto random_sv = createRandomState<PrecisionT>(re, num_qubits);
        init_sv = random_sv;
    }
    auto local_state = mpi_manager.scatter(init_sv, 0);
    mpi_manager.Barrier();
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                    nGlobalIndexBits, nLocalIndexBits);
    sv.CopyHostDataToGpu(local_state, false);
    PrecisionT mpi_result, result;
    std::vector<size_t> wire{0};

    std::vector<cp_t> matrix(4, cp_t{0.0, 0.0});

    matrix[0] = cp_t{1.0, 0.0};
    matrix[3] = cp_t{-1.0, 0.0};

    mpi_result = sv.expval(wire, matrix).x;

    SVDataGPU<TestType> svdat{num_qubits, init_sv};
    if (mpi_manager.getRank() == 0) {
        result = svdat.cuda_sv.expval(wire, matrix).x;
    }
    mpi_manager.Bcast<PrecisionT>(result, 0);
    CHECK(result == Approx(mpi_result).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::expval_Hardmard",
                   "[StateVectorCudaMPI_Nonparam]", double) {
    using cp_t = std::complex<TestType>;
    using PrecisionT = TestType;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    size_t mpi_buffersize = num_qubits;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t svLength = 1 << num_qubits;
    mpi_manager.Barrier();
    std::vector<cp_t> init_sv(svLength);
    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        auto random_sv = createRandomState<PrecisionT>(re, num_qubits);
        init_sv = random_sv;
    }
    auto local_state = mpi_manager.scatter(init_sv, 0);
    mpi_manager.Barrier();
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                    nGlobalIndexBits, nLocalIndexBits);
    sv.CopyHostDataToGpu(local_state, false);
    PrecisionT mpi_result, result;
    std::vector<size_t> wire{0};

    std::vector<cp_t> matrix(4, cp_t{1.0 / sqrt(2.0), 0.0});

    matrix[3] = -matrix[3];

    mpi_result = sv.expval(wire, matrix).x;

    SVDataGPU<TestType> svdat{num_qubits, init_sv};
    if (mpi_manager.getRank() == 0) {
        result = svdat.cuda_sv.expval(wire, matrix).x;
    }
    mpi_manager.Bcast<PrecisionT>(result, 0);
    CHECK(result == Approx(mpi_result).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::probability",
                   "[StateVectorCudaMPI_Nonparam]", double) {
    using cp_t = std::complex<TestType>;
    const std::size_t numqubits = 4;
    MPIManager mpi_manager(MPI_COMM_WORLD);
    size_t mpi_buffersize = num_qubits;
    size_t nGlobalIndexBits = std::bit_width(mpi_manager.getSize()) - 1;
    size_t nLocalIndexBits = numqubits - nGlobalIndexBits;

    std::vector<cp_t> init_sv{{0.1653855288944372, 0.08360762242222763},
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

    auto local_state = mpi_manager.scatter(init_sv, 0);

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("Subset probability at global wires") {

        std::vector<std::size_t> wires = {0, 1};

        StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                        nGlobalIndexBits, nLocalIndexBits);

        sv.CopyHostDataToGpu(local_state, false);

        auto probs = sv.probability(wires);

        std::vector<double> expected = {0.26471457, 0.15697763, 0.31723892,
                                        0.26106889};

        auto local_expected = mpi_manager.scatter(expected, 0);

        CHECK(local_expected == Pennylane::approx(probs));
    }

    SECTION("Subset probability at both local and global wires") {
        std::vector<std::size_t> wires = {1, 3};

        StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                        nGlobalIndexBits, nLocalIndexBits);
        sv.CopyHostDataToGpu(local_state, false);

        auto probs = sv.probability(wires);

        std::vector<double> expected = {0.38201245, 0.19994104, 0.16270186,
                                        0.25534466};
        // tranposed
        std::vector<double> expected_rank0 = {0.38201245, 0.16270186,
                                              0.19994104, 0.25534466};
        std::vector<double> expected_rank1 = {};
        if (mpi_manager.getRank() == 0) {
            CHECK(expected_rank0 == Pennylane::approx(probs));
        } else if (mpi_manager.getRank() == 1) {
            CHECK(expected_rank1 == Pennylane::approx(probs));
        }
    }

    SECTION("Subset probability at local wires") {
        std::vector<std::size_t> wires = {3};

        StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                        nGlobalIndexBits, nLocalIndexBits);

        sv.CopyHostDataToGpu(local_state, false);

        auto probs = sv.probability(wires);

        std::vector<double> expected_rank0 = {0.54471431, 0.45528569};
        std::vector<double> expected_nonrank0 = {};
        if (mpi_manager.getRank() == 0) {
            CHECK(expected_rank0 == Pennylane::approx(probs));
        } else {
            CHECK(expected_nonrank0 == Pennylane::approx(probs));
        }
    }
}

TEMPLATE_TEST_CASE("Sample", "[LightningGPUMPI_NonParam]", double) {
    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    using cp_t = std::complex<TestType>;
    const std::size_t numqubits = 4;
    MPIManager mpi_manager(MPI_COMM_WORLD);
    size_t mpi_buffersize = 26;
    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = numqubits - nGlobalIndexBits;

    std::vector<cp_t> init_sv{{0.1653855288944372, 0.08360762242222763},
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

    auto local_state = mpi_manager.scatter(init_sv, 0);

    std::vector<TestType> expected_probabilities = {
        0.03434261, 0.02279588, 0.12467259, 0.08290349, 0.053264,   0.02578699,
        0.03568799, 0.04223866, 0.04648469, 0.06486717, 0.17651256, 0.0293745,
        0.05134485, 0.09621607, 0.02240502, 0.09110293};

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    StateVectorCudaMPI<TestType> sv(mpi_manager, dt_local, mpi_buffersize,
                                    nGlobalIndexBits, nLocalIndexBits);
    sv.CopyHostDataToGpu(local_state, false);

    size_t N = std::pow(2, numqubits);
    size_t num_samples = 1000;

    auto &&samples = sv.generate_samples(num_samples);

    std::vector<size_t> counts(N, 0);
    std::vector<size_t> samples_decimal(num_samples, 0);

    // convert samples to decimal and then bin them in counts
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < numqubits; j++) {
            if (samples[i * numqubits + j] != 0) {
                samples_decimal[i] += twos[(numqubits - 1 - j)];
            }
        }
        counts[samples_decimal[i]] += 1;
    }
    // compute estimated probabilities from histogram
    std::vector<TestType> probabilities(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (TestType)num_samples;
    }

    REQUIRE_THAT(probabilities,
                 Catch::Approx(expected_probabilities).margin(.05));
}