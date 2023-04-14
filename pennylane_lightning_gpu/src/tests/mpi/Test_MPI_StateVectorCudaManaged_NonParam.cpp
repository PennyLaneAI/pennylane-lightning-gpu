#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>
#include <mpi.h>

#include "StateVectorCudaMPI.hpp"
#include "StateVectorRawCPU.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "../TestHelpers.hpp"

using namespace Pennylane;
using namespace CUDA;

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SetStateVector",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using PrecisionT = TestType;
    const std::size_t num_qubits = 5;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype message_type;
    MPI_Datatype message_int_type;

    if constexpr (std::is_same_v<PrecisionT, double>) {
        message_type = MPI_DOUBLE_COMPLEX;
        message_int_type = MPI_INT64_T;
    } else {
        message_type = MPI_COMPLEX;
        message_int_type = MPI_INT32_T;
    }

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < size) {
        ++nGlobalIndexBits;
    }

    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<std::complex<PrecisionT>> init_state(
        Pennylane::Util::exp2(num_qubits));
    std::vector<std::complex<PrecisionT>> expected_state(
        Pennylane::Util::exp2(num_qubits));
    std::vector<std::complex<PrecisionT>> local_state(subSvLength);
    std::vector<std::complex<PrecisionT>> expected_local_state(subSvLength);

    using index_type =
        typename std::conditional<std::is_same<PrecisionT, float>::value,
                                  int32_t, int64_t>::type;

    std::vector<index_type> indices(Pennylane::Util::exp2(num_qubits));

    if (rank == 0) {
        std::mt19937 re{1337};
        init_state = createRandomState<PrecisionT>(re, num_qubits);
        expected_state = init_state;
        for (size_t i = 0; i < Pennylane::Util::exp2(num_qubits - 1); i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
            indices[i * 2] = i * 2 + 1;
            indices[i * 2 + 1] = i * 2;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(expected_state.data(), subSvLength, message_type,
                expected_local_state.data(), subSvLength, message_type, 0,
                MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(indices.data(), indices.size(), message_int_type, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(init_state.data(), init_state.size(), message_type, 0,
              MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        int nDevices = 0; // Number of GPU devices per node
        cudaGetDeviceCount(&nDevices);
        int deviceId = rank % nDevices;
        cudaSetDevice(deviceId);
        StateVectorCudaMPI<PrecisionT> sv(MPI_COMM_WORLD, nLocalIndexBits);
        MPI_Barrier(MPI_COMM_WORLD);

        // The setStates will shuffle the state vector values on the device with
        // the following indices and values setting on host. For example, the
        // values[i] is used to set the indices[i] th element of state vector on
        // the device. For example, values[2] (init_state[5]) will be copied to
        // indices[2]th or (4th) element of the state vector.

        sv.template setStateVector<index_type>(
            init_state.size(), init_state.data(), indices.data(), false);
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(local_state.data(),
                             static_cast<std::size_t>(subSvLength));
        MPI_Barrier(MPI_COMM_WORLD);

        CHECK(expected_local_state == Pennylane::approx(local_state));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SetIthStates",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using PrecisionT = TestType;
    const std::size_t num_qubits = 5;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype message_type;
    if constexpr (std::is_same_v<PrecisionT, double>) {
        message_type = MPI_DOUBLE_COMPLEX;
    } else {
        message_type = MPI_COMPLEX;
    }

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < size) {
        ++nGlobalIndexBits;
    }

    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

    int index;
    if (rank == 0) {
        std::mt19937 re{1337};
        std::uniform_int_distribution<> distr(
            0, Pennylane::Util::exp2(num_qubits) - 1);
        index = distr(re);
    }
    MPI_Bcast(&index, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::complex<PrecisionT>> expected_state(
        Pennylane::Util::exp2(num_qubits), {0, 0});
    if (rank == 0) {
        expected_state[index] = {1.0, 0};
    }

    std::vector<std::complex<PrecisionT>> expected_local_state(
        Pennylane::Util::exp2(nLocalIndexBits), {0, 0});

    MPI_Scatter(expected_state.data(), expected_local_state.size(),
                message_type, expected_local_state.data(),
                expected_local_state.size(), message_type, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        int nDevices = 0; // Number of GPU devices per node
        cudaGetDeviceCount(&nDevices);
        int deviceId = rank % nDevices;
        cudaSetDevice(deviceId);

        StateVectorCudaMPI<PrecisionT> sv(MPI_COMM_WORLD, nLocalIndexBits);
        std::complex<PrecisionT> values = {1.0, 0};
        sv.setBasisState(values, index, false);

        int subSvLength = 1 << nLocalIndexBits;
        std::vector<std::complex<TestType>> h_sv0(subSvLength, {0.0, 0.0});
        sv.CopyGpuDataToHost(h_sv0.data(),
                             static_cast<std::size_t>(subSvLength));

        CHECK(expected_local_state == Pennylane::approx(h_sv0));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyPauliX",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype message_type;

    const std::size_t num_qubits = 4;

    if constexpr (std::is_same_v<TestType, double>) {
        message_type = MPI_DOUBLE_COMPLEX;
    } else {
        message_type = MPI_COMPLEX;
    }

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < size) {
        ++nGlobalIndexBits;
    }

    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<cp_t> h_sv = {{0.0, 0.0}, {0.1, 0.0}, {0.2, 0.0}, {0.3, 0.0},
                              {0.4, 0.0}, {0.5, 0.0}, {0.6, 0.0}, {0.7, 0.0},
                              {0.8, 0.0}, {0.9, 0.0}, {1.0, 0.0}, {1.1, 0.0},
                              {1.2, 0.0}, {1.3, 0.0}, {1.4, 0.0}, {1.5, 0.0}};

    std::vector<std::vector<cp_t>> h_sv_ref{std::vector<cp_t>{{0.8, 0.0},
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

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            int deviceId = rank % nDevices;
            cudaSetDevice(deviceId);

            StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            int subSvLength = 1 << nLocalIndexBits;
            std::vector<std::complex<TestType>> h_sv0(subSvLength, {0.0, 0.0});
            MPI_Scatter(h_sv.data(), subSvLength, message_type, h_sv0.data(),
                        subSvLength, message_type, 0, MPI_COMM_WORLD);

            sv.CopyHostDataToGpu(h_sv0, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.applyPauliX({0}, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.CopyGpuDataToHost(h_sv0.data(),
                                 static_cast<std::size_t>(subSvLength));
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Allgather(h_sv0.data(), subSvLength, message_type, h_sv.data(),
                          subSvLength, message_type, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            CHECK(h_sv == Pennylane::approx(h_sv_ref[0]));
        }

        SECTION("Operation on localQubits") {
            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            int deviceId = rank % nDevices;
            cudaSetDevice(deviceId);

            StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            int subSvLength = 1 << nLocalIndexBits;
            std::vector<std::complex<TestType>> h_sv0(subSvLength, {0.0, 0.0});

            MPI_Scatter(h_sv.data(), subSvLength, message_type, h_sv0.data(),
                        subSvLength, message_type, 0, MPI_COMM_WORLD);

            sv.CopyHostDataToGpu(h_sv0, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.applyPauliX({num_qubits - 1}, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.CopyGpuDataToHost(h_sv0.data(),
                                 static_cast<std::size_t>(subSvLength));
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Allgather(h_sv0.data(), subSvLength, message_type, h_sv.data(),
                          subSvLength, message_type, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            CHECK(h_sv == Pennylane::approx(h_sv_ref[1]));
        }
    }

    SECTION("Apply using dispatcher") {
        SECTION("Operation on globalQubits") {
            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            int deviceId = rank % nDevices;
            cudaSetDevice(deviceId);

            StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            int subSvLength = 1 << nLocalIndexBits;
            std::vector<std::complex<TestType>> h_sv0(subSvLength, {0.0, 0.0});
            MPI_Scatter(h_sv.data(), subSvLength, message_type, h_sv0.data(),
                        subSvLength, message_type, 0, MPI_COMM_WORLD);

            sv.CopyHostDataToGpu(h_sv0, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.applyPauliX({0}, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.CopyGpuDataToHost(h_sv0.data(),
                                 static_cast<std::size_t>(subSvLength));
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Allgather(h_sv0.data(), subSvLength, message_type, h_sv.data(),
                          subSvLength, message_type, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            CHECK(h_sv == Pennylane::approx(h_sv_ref[0]));
        }

        SECTION("Operation on localQubits") {
            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            int deviceId = rank % nDevices;
            cudaSetDevice(deviceId);

            StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            int subSvLength = 1 << nLocalIndexBits;
            std::vector<std::complex<TestType>> h_sv0(subSvLength, {0.0, 0.0});

            MPI_Scatter(h_sv.data(), subSvLength, message_type, h_sv0.data(),
                        subSvLength, message_type, 0, MPI_COMM_WORLD);

            sv.CopyHostDataToGpu(h_sv0, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.applyPauliX({num_qubits - 1}, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.CopyGpuDataToHost(h_sv0.data(),
                                 static_cast<std::size_t>(subSvLength));
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Allgather(h_sv0.data(), subSvLength, message_type, h_sv.data(),
                          subSvLength, message_type, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            CHECK(h_sv == Pennylane::approx(h_sv_ref[1]));
        }
    }
}


TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyPauliY",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    using PrecisionT = TestType;
    const std::size_t num_qubits = 4;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype message_type;

    if constexpr (std::is_same_v<TestType, double>) {
        message_type = MPI_DOUBLE_COMPLEX;
    } else {
        message_type = MPI_COMPLEX;
    }

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < size) {
        ++nGlobalIndexBits;
    }

    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

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
        std::vector<cp_t>{
            {0.12834321, -0.17324176}, {0.25377766, -0.02154223},
            {0.30227665, -0.29178997}, {0.01388092, -0.17082688},
            {0.22338163, -0.03801974}, {0.23785467, -0.19910106},
            {0.05717371, -0.13833362}, {0.22946371, -0.19608503},
            {-0.08360762, 0.16538553}, {-0.13209081, 0.07312934},
            {-0.26134408, 0.23742759}, {-0.23406072, 0.16768741},
            {-0.05246906, 0.22474651}, {-0.01835598, 0.15953071},
            {-0.18836803, 0.01433429}, {-0.02069818, 0.20447554}}
    };

    std::vector<std::vector<cp_t>> init_localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));
    std::vector<std::vector<cp_t>> expected_localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));
        std::vector<std::vector<cp_t>> localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    MPI_Scatter(initstate[0].data(), init_localstate[0].size(), message_type,
                init_localstate[0].data(), init_localstate[0].size(),
                message_type, 0, MPI_COMM_WORLD);
    
    MPI_Scatter(expected_results[0].data(), expected_localstate[0].size(),
                message_type, expected_localstate[0].data(),
                expected_localstate[0].size(), message_type, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    SECTION("Apply directly") {
        SECTION("Operation on globalQubits") {
            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            int deviceId = rank % nDevices;
            cudaSetDevice(deviceId);
            StateVectorCudaMPI<PrecisionT> sv(MPI_COMM_WORLD, nLocalIndexBits);

            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.CopyHostDataToGpu(init_localstate[0], false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.applyPauliY({0}, false);
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            sv.CopyGpuDataToHost(localstate[0].data(),
                             static_cast<std::size_t>(subSvLength));
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            CHECK(localstate[0] == Pennylane::approx(expected_localstate[0]));
        }
    }

}


TEMPLATE_TEST_CASE("StateVectorCudaMPI::Hamiltonian_expval",
                   "[StateVectorCudaMPI_Nonparam]", double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype message_type;

    if constexpr (std::is_same_v<TestType, double>) {
        message_type = MPI_DOUBLE_COMPLEX;
    } else {
        message_type = MPI_COMPLEX;
    }

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < size) {
        ++nGlobalIndexBits;
    }

    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

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

    std::vector<std::vector<cp_t>> init_localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    MPI_Scatter(initstate[0].data(), init_localstate[0].size(), message_type,
                init_localstate[0].data(), init_localstate[0].size(),
                message_type, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    SECTION("GetExpectionIdentity") {
        int nDevices = 0; // Number of GPU devices per node
        cudaGetDeviceCount(&nDevices);
        int deviceId = rank % nDevices;
        cudaSetDevice(deviceId);

        StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyHostDataToGpu(init_localstate[0], false);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.applyHadamard({0}, false);
        sv.applyCNOT({0, 1}, false);
        sv.applyCNOT({2, 3}, false);
        MPI_Barrier(MPI_COMM_WORLD);

        size_t matrix_dim = 2;
        std::vector<cp_t> matrix(matrix_dim * matrix_dim);

        for (size_t i = 0; i < matrix.size(); i++) {
            if (i == 0 || i == 3)
                matrix[i] = std::complex<TestType>(0, 0);
            else
                matrix[i] = std::complex<TestType>(1, 0);
        }
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto results = sv.expval({3}, matrix);

        cp_t expected(0.7181199813879771, 0);
        CHECK(expected.real() == Approx(results.x).epsilon(1e-7));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::probability",
                   "[StateVectorCudaMPI_Nonparam]", double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype message_type;

    if constexpr (std::is_same_v<TestType, double>) {
        message_type = MPI_DOUBLE_COMPLEX;
    } else {
        message_type = MPI_COMPLEX;
    }

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < size) {
        ++nGlobalIndexBits;
    }

    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

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

    std::vector<std::vector<cp_t>> init_localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    MPI_Scatter(initstate[0].data(), init_localstate[0].size(), message_type,
                init_localstate[0].data(), init_localstate[0].size(),
                message_type, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    SECTION("Subset probability at global wires") {

        std::vector<std::size_t> wires = {0, 1};

        int nDevices = 0; // Number of GPU devices per node
        cudaGetDeviceCount(&nDevices);
        int deviceId = rank % nDevices;
        cudaSetDevice(deviceId);

        StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyHostDataToGpu(init_localstate[0], false);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        auto probs = sv.probability(wires);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<double> expected = {0.26471457, 0.15697763, 0.31723892,
                                        0.26106889};
        CHECK(expected == Pennylane::approx(probs));
    }

    SECTION("Subset probability at both local and global wires") {
        std::vector<std::size_t> wires = {1, 3};

        int nDevices = 0; // Number of GPU devices per node
        cudaGetDeviceCount(&nDevices);
        int deviceId = rank % nDevices;
        cudaSetDevice(deviceId);

        StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyHostDataToGpu(init_localstate[0], false);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        auto probs = sv.probability(wires);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<double> expected = {0.38201245, 0.19994104, 0.16270186,
                                        0.25534466};
        CHECK(expected == Pennylane::approx(probs));
    }

    SECTION("Subset probability at local wires") {
        std::vector<std::size_t> wires = {3};

        int nDevices = 0; // Number of GPU devices per node
        cudaGetDeviceCount(&nDevices);
        int deviceId = rank % nDevices;
        cudaSetDevice(deviceId);

        StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyHostDataToGpu(init_localstate[0], false);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        auto probs = sv.probability(wires);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<double> expected = {0.54471431, 0.45528569};
        CHECK(expected == Pennylane::approx(probs));
    }
}

TEMPLATE_TEST_CASE("Sample", "[LightningGPU_NonParam]", double) {
    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 4;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype message_type;

    if constexpr (std::is_same_v<TestType, double>) {
        message_type = MPI_DOUBLE_COMPLEX;
    } else {
        message_type = MPI_COMPLEX;
    }

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < size) {
        ++nGlobalIndexBits;
    }

    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    int subSvLength = 1 << nLocalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

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

    std::vector<std::vector<cp_t>> init_localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    MPI_Scatter(initstate[0].data(), init_localstate[0].size(), message_type,
                init_localstate[0].data(), init_localstate[0].size(),
                message_type, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Defining the State Vector that will be measured.

    SECTION("Subset probability") {
        int nDevices = 0; // Number of GPU devices per node
        cudaGetDeviceCount(&nDevices);
        int deviceId = rank % nDevices;
        cudaSetDevice(deviceId);

        StateVectorCudaMPI<TestType> sv(MPI_COMM_WORLD, nLocalIndexBits);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyHostDataToGpu(init_localstate[0], false);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<TestType> expected_probabilities = {
            0.03434261, 0.02279588, 0.12467259, 0.08290349,
            0.053264,   0.02578699, 0.03568799, 0.04223866,
            0.04648469, 0.06486717, 0.17651256, 0.0293745,
            0.05134485, 0.09621607, 0.02240502, 0.09110293};

        size_t N = std::pow(2, num_qubits);
        size_t num_samples = 1000;

        auto &&samples = sv.generate_samples(num_samples);

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

        REQUIRE_THAT(probabilities,
                     Catch::Approx(expected_probabilities).margin(.05));
    }
}
