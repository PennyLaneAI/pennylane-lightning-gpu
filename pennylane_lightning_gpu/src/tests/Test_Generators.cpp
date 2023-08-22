#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <catch2/catch.hpp>

#include "GateGenerators.hpp"
#include "StateVectorCudaManaged.hpp"
#include "TestHelpersLGPU.hpp"

using namespace Pennylane::CUDA;
using namespace Pennylane::CUDA::Generators;
using namespace Pennylane::Algorithms;

// NOTE: the scaling factors are implicitly included in the Adjoint Jacobian
// evaluation, so excluded from the matrices here.

TEST_CASE("Generators::applyGeneratorRX_GPU", "[GateGenerators]") {
    // grad(RX) = grad(e^{-i*0.5*PauliX*a}) => -i*0.5*PauliX
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        cuGates::getPauliX<typename StateVectorCudaManaged<double>::CFP_t>()};
    std::mt19937 re{1337U};
    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (std::size_t applied_qubit = 0; applied_qubit < num_qubits;
             applied_qubit++) {
            SVDataGPU<double> psi(num_qubits);
            psi.sv.updateData(createRandomState<double>(re, num_qubits));
            psi.cuda_sv.CopyHostDataToGpu(psi.sv);

            SVDataGPU<double> psi_direct(num_qubits);
            psi_direct.sv.updateData(psi.sv.getDataVector());
            psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

            std::string cache_gate_name = "DirectGenRX" +
                                          std::to_string(applied_qubit) + "_" +
                                          std::to_string(num_qubits);

            applyGeneratorRX_GPU(psi.cuda_sv, {applied_qubit}, false);
            psi_direct.cuda_sv.applyOperation(cache_gate_name, {applied_qubit},
                                              false, {0.0}, matrix);
            psi.cuda_sv.CopyGpuDataToHost(psi.sv);
            psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
            CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
        }
    }
}

TEST_CASE("Generators::applyGeneratorRY_GPU", "[GateGenerators]") {
    // grad(RY) = grad(e^{-i*0.5*PauliY*a}) => -i*0.5*PauliY
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        cuGates::getPauliY<typename StateVectorCudaManaged<double>::CFP_t>()};
    std::mt19937 re{1337U};
    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (std::size_t applied_qubit = 0; applied_qubit < num_qubits;
             applied_qubit++) {
            SVDataGPU<double> psi(num_qubits);
            psi.sv.updateData(createRandomState<double>(re, num_qubits));
            psi.cuda_sv.CopyHostDataToGpu(psi.sv);

            SVDataGPU<double> psi_direct(num_qubits);
            psi_direct.sv.updateData(psi.sv.getDataVector());
            psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

            std::string cache_gate_name = "DirectGenRY" +
                                          std::to_string(applied_qubit) + "_" +
                                          std::to_string(num_qubits);

            applyGeneratorRY_GPU(psi.cuda_sv, {applied_qubit}, false);
            psi_direct.cuda_sv.applyOperation(cache_gate_name, {applied_qubit},
                                              false, {0.0}, matrix);
            psi.cuda_sv.CopyGpuDataToHost(psi.sv);
            psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
            CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
        }
    }
}

TEST_CASE("Generators::applyGeneratorRZ_GPU", "[GateGenerators]") {
    // grad(RZ) = grad(e^{-i*0.5*PauliZ*a}) => -i*0.5*PauliZ
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        cuGates::getPauliZ<typename StateVectorCudaManaged<double>::CFP_t>()};
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (std::size_t applied_qubit = 0; applied_qubit < num_qubits;
             applied_qubit++) {
            SVDataGPU<double> psi(num_qubits);
            psi.sv.updateData(createRandomState<double>(re, num_qubits));
            psi.cuda_sv.CopyHostDataToGpu(psi.sv);

            SVDataGPU<double> psi_direct(num_qubits);
            psi_direct.sv.updateData(psi.sv.getDataVector());
            psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

            std::string cache_gate_name = "DirectGenRZ" +
                                          std::to_string(applied_qubit) + "_" +
                                          std::to_string(num_qubits);

            applyGeneratorRZ_GPU(psi.cuda_sv, {applied_qubit}, false);
            psi_direct.cuda_sv.applyOperation(cache_gate_name, {applied_qubit},
                                              false, {0.0}, matrix);
            psi.cuda_sv.CopyGpuDataToHost(psi.sv);
            psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
            CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
        }
    }
}

TEST_CASE("Generators::applyGeneratorPhaseShift_GPU", "[GateGenerators]") {
    // grad(PhaseShift) = grad(e^{i*0.5*a}*e^{-i*0.5*PauliZ*a}) => -i|1><1|
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (std::size_t applied_qubit = 0; applied_qubit < num_qubits;
             applied_qubit++) {

            SVDataGPU<double> psi(num_qubits);
            psi.sv.updateData(createRandomState<double>(re, num_qubits));
            psi.cuda_sv.CopyHostDataToGpu(psi.sv);

            SVDataGPU<double> psi_direct(num_qubits);
            psi_direct.sv.updateData(psi.sv.getDataVector());
            psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

            std::string cache_gate_name = "DirectGenPhaseShift" +
                                          std::to_string(applied_qubit) + "_" +
                                          std::to_string(num_qubits);

            applyGeneratorPhaseShift_GPU(psi.cuda_sv, {applied_qubit}, false);
            psi_direct.cuda_sv.applyOperation(cache_gate_name, {applied_qubit},
                                              false, {0.0}, matrix);
            psi.cuda_sv.CopyGpuDataToHost(psi.sv);
            psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
            CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
        }
    }
}

TEST_CASE("Generators::applyGeneratorIsingXX_GPU", "[GateGenerators]") {
    // grad(IsingXX)() = e^{-i*0.5*a*(kron(X, X))}) => -0.5*i*(kron(X, X))
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {

                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenIsingXX" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorIsingXX_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenIsingXX" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorIsingXX_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorIsingYY_GPU", "[GateGenerators]") {
    // grad(IsingXX)() = e^{-i*0.5*a*(kron(Y, Y))}) => -0.5*i*(kron(Y, Y))
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenIsingYY" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorIsingYY_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenIsingYY" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorIsingYY_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorIsingZZ_GPU", "[GateGenerators]") {
    // grad(IsingXX)() = e^{-i*0.5*a*(kron(Z, Z))}) => -0.5*i*(kron(Z, Z))
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenIsingZZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorIsingZZ_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenIsingZZ" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorIsingZZ_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

///////////////////////////////////////

TEST_CASE("Generators::applyGeneratorCRX_GPU", "[GateGenerators]") {
    // grad(CRX) = grad(kron(|0><0|, I(2)) + kron(|1><1|,
    // e^{-i*0.5*(PauliX)*a})) => -i*0.5*kron(|1><1|, PauliX)
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenCRX" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorCRX_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenCRX" + std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorCRX_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorCRY_GPU", "[GateGenerators]") {
    // grad(CRY) = grad(kron(|0><0|, I(2)) + kron(|1><1|,
    // e^{-i*0.5*(PauliY)*a})) => -i*0.5*kron(|1><1|, PauliY)
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenCRY" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorCRY_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenCRY" + std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorCRY_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorCRZ_GPU", "[GateGenerators]") {
    // grad(CRZ) = grad(kron(|0><0|, I(2)) + kron(|1><1|,
    // e^{-i*0.5*(PauliZ)*a})) => -i*0.5*kron(|1><1|, PauliZ)
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenCRZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorCRZ_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenCRZ" + std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorCRZ_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorControlledPhaseShift_GPU",
          "[GateGenerators]") {
    // grad(ControlledPhaseShift) = grad(kron(|0><0|, I(2)) +  kron(|1><1|,
    // e^{i*0.5*a}*e^{-i*0.5*PauliZ*a} )) => -i|11><11|
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenControlledPhaseShift" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorControlledPhaseShift_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenControlledPhaseShift" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorControlledPhaseShift_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorSingleExcitation_GPU",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenSingleExcitation" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorSingleExcitation_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenSingleExcitation" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorSingleExcitation_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorSingleExcitationMinus_GPU",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0,-1.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenSingleExcitationMinus" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorSingleExcitationMinus_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenSingleExcitationMinus" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorSingleExcitationMinus_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorSingleExcitationPlus_GPU",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {-1.0, 0.0},{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0,-1.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenSingleExcitationPlus" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorSingleExcitationPlus_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenSingleExcitationPlus" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorSingleExcitationPlus_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorDoubleExcitation_GPU",
          "[GateGenerators]") {
    // clang-format off
    /* For convenience, the DoubleExcitation* matrices were generated from PennyLane and formatted as follows:
        ```python
            mat = qml.matrix(qml.DoubleExcitation(a, wires=[0,1,2,3]).generator())
            def cpp_format(arr):
                s = ""
                for i in arr:
                    s += f"{{{np.real(i) if np.real(i) != 0 else 0}, {np.imag(i) if np.imag(i) != 0 else 0}}},"
                return s
            output = ""

            for i in range(mat.shape[0]):
                out += cpp_format(mat[i][:])
                out += "\n"
            print(output)
        ```
    */
    // clang-format on

    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, -1.0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 1.0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 4; num_qubits <= 8; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenDoubleExcitation" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorDoubleExcitation_GPU(
                    psi.cuda_sv,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false, {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenDoubleExcitation" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorDoubleExcitation_GPU(
                    psi.cuda_sv,
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false, {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorDoubleExcitationMinus_GPU",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, -1.0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 1.0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 4; num_qubits <= 8; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenDoubleExcitationMinus" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorDoubleExcitationMinus_GPU(
                    psi.cuda_sv,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false, {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenDoubleExcitationMinus" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorDoubleExcitationMinus_GPU(
                    psi.cuda_sv,
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false, {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorDoubleExcitationPlus_GPU",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, -1.0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 1.0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 4; num_qubits <= 8; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenDoubleExcitationPlus" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorDoubleExcitationPlus_GPU(
                    psi.cuda_sv,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false, {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenDoubleExcitationPlus" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorDoubleExcitationPlus_GPU(
                    psi.cuda_sv,
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false, {0.0}, matrix);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorMultiRZ_GPU", "[GateGenerators]") {

    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix2{
        // clang-format off
        {1.0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{-1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{-1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{1.0, 0}
        // clang-format on
    };

    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix3{
        // clang-format off
        {1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0}
        // clang-format on
    };

    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix4{
        // clang-format off
        {1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 4; num_qubits <= 8; num_qubits++) {
        SECTION("Increasing qubit indices, MultiRZ 2") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorMultiRZ_GPU(
                    psi.cuda_sv, {applied_qubit, applied_qubit + 1}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit, applied_qubit + 1}, false,
                    {0.0}, matrix2);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices, MultiRZ 2") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorMultiRZ_GPU(
                    psi.cuda_sv, {applied_qubit + 1, applied_qubit}, false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name, {applied_qubit + 1, applied_qubit}, false,
                    {0.0}, matrix2);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }

        SECTION("Increasing qubit indices, MultiRZ 3") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 2;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorMultiRZ_GPU(
                    psi.cuda_sv,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2},
                    false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2},
                    false, {0.0}, matrix3);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices, MultiRZ 3") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 2;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit + 2) +
                    "_" + std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorMultiRZ_GPU(
                    psi.cuda_sv,
                    {applied_qubit + 2, applied_qubit + 1, applied_qubit},
                    false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit + 2, applied_qubit + 1, applied_qubit},
                    false, {0.0}, matrix3);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }

        SECTION("Increasing qubit indices, MultiRZ 4") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorMultiRZ_GPU(psi.cuda_sv,
                                          {applied_qubit, applied_qubit + 1,
                                           applied_qubit + 2,
                                           applied_qubit + 3},
                                          false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false, {0.0}, matrix4);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                SVDataGPU<double> psi(num_qubits);
                psi.sv.updateData(createRandomState<double>(re, num_qubits));
                psi.cuda_sv.CopyHostDataToGpu(psi.sv);

                SVDataGPU<double> psi_direct(num_qubits);
                psi_direct.sv.updateData(psi.sv.getDataVector());
                psi_direct.cuda_sv.CopyHostDataToGpu(psi.sv);

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit + 3) +
                    "_" + std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                applyGeneratorMultiRZ_GPU(psi.cuda_sv,
                                          {applied_qubit + 3, applied_qubit + 2,
                                           applied_qubit + 1, applied_qubit},
                                          false);
                psi_direct.cuda_sv.applyOperation(
                    cache_gate_name,
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false, {0.0}, matrix4);
                psi.cuda_sv.CopyGpuDataToHost(psi.sv);
                psi_direct.cuda_sv.CopyGpuDataToHost(psi_direct.sv);
                CHECK(psi.sv.getDataVector() == psi_direct.sv.getDataVector());
            }
        }
    }
}