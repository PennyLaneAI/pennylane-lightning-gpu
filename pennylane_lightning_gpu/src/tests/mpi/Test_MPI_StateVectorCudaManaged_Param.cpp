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
#include "StateVectorRawCPU.hpp"

#include "MPIManager.hpp"

#include "../TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::MPI;
using namespace CUDA;

/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRX", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < mpi_manager.getSize()) {
        ++nGlobalIndexBits;
    }

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

    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices));
    int deviceId = mpi_manager.getRank() % nDevices;
    PL_CUDA_IS_SUCCESS(cudaSetDevice(deviceId));

    SECTION("Apply directly at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyRX({0}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyRX({num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("RX", {0}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("RX", {num_qubits - 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRY", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < mpi_manager.getSize()) {
        ++nGlobalIndexBits;
    }

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
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyRY({0}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyRY({num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("RY", {0}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("RY", {num_qubits - 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRZ", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < mpi_manager.getSize()) {
        ++nGlobalIndexBits;
    }

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
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyRZ({0}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyRZ({num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("RZ", {0}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("RZ", {num_qubits - 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyPhaseShift",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < mpi_manager.getSize()) {
        ++nGlobalIndexBits;
    }

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
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyPhaseShift({0}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyPhaseShift({num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("PhaseShift", {0}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("PhaseShift", {num_qubits - 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRot", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < mpi_manager.getSize()) {
        ++nGlobalIndexBits;
    }

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
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyRot({0}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyRot({num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at a global wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("Rot", {0}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at a local wire") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("Rot", {num_qubits - 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }
}

/*******************************************************************************
 * Two-qubit gates
 ******************************************************************************/
// IsingXX Gate
TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyIsingXX",
                   "[StateVectorCudaMPI_Param]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < mpi_manager.getSize()) {
        ++nGlobalIndexBits;
    }

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
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingXX({0, 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingXX({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingXX({1, num_qubits - 2}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingXX", {0, 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingXX", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingXX", {1, num_qubits - 2}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
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

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < mpi_manager.getSize()) {
        ++nGlobalIndexBits;
    }

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
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingYY({0, 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingYY({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingYY({1, num_qubits - 2}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingYY", {0, 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingYY", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingYY", {1, num_qubits - 2}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
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

    int nGlobalIndexBits = 0;
    while ((1 << nGlobalIndexBits) < mpi_manager.getSize()) {
        ++nGlobalIndexBits;
    }

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
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingZZ({0, 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply directly at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingZZ({num_qubits - 2, num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply directly at both local and global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyIsingZZ({1, num_qubits - 2}, false, angles[0]);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }

    SECTION("Apply using dispatcher at global wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingZZ", {0, 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate0));
    }

    SECTION("Apply using dispatcher at local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingZZ", {num_qubits - 2, num_qubits - 1}, false,
                          {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate1));
    }

    SECTION("Apply using dispatcher at both global and local wires") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, nGlobalIndexBits,
                                          nLocalIndexBits);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyHostDataToGpu(init_localstate, false);
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.applyOperation("IsingZZ", {1, num_qubits - 2}, false, {angles[0]});
        cudaDeviceSynchronize();
        mpi_manager.Barrier();

        sv.CopyGpuDataToHost(localstate.data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        mpi_manager.Barrier();
        CHECK(localstate == Pennylane::approx(expected_localstate2));
    }
}

// ControlledPhaseShift Gate

// CRX Gate

// CRY Gate

// CRZ Gate

// CRot Gate

// SingleExcitation Gate

// SingleExcitationMinus Gate

// SingleExcitationPlus Gate

// DoubleExcitation Gate

// DoubleExcitationMinus Gate

// DoubleExcitationPlus Gate

// MultiRZ Gate
