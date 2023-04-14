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

#include "../TestHelpers.hpp"

using namespace Pennylane;
using namespace CUDA;

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRX", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

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
    int subSvLength = 1 << nLocalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

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

    std::vector<std::vector<cp_t>> init_localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    std::vector<std::vector<cp_t>> expected_localstate(
        2, std::vector<cp_t>(subSvLength, {0.0, 0.0}));
    std::vector<std::vector<cp_t>> localstate(
        2, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    MPI_Scatter(initstate[0].data(), init_localstate[0].size(), message_type,
                init_localstate[0].data(), init_localstate[0].size(),
                message_type, 0, MPI_COMM_WORLD);
    MPI_Scatter(expected_results[0].data(), expected_localstate[0].size(),
                message_type, expected_localstate[0].data(),
                expected_localstate[0].size(), message_type, 0, MPI_COMM_WORLD);
    MPI_Scatter(expected_results[1].data(), expected_localstate[1].size(),
                message_type, expected_localstate[1].data(),
                expected_localstate[1].size(), message_type, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    SECTION("Apply directly at a global wire") {
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

        sv.applyRX({0}, false, angles[0]);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[0].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[0] == Pennylane::approx(expected_localstate[0]));
    }

    SECTION("Apply directly at a local wire") {
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

        sv.applyRX({num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[1].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[1] == Pennylane::approx(expected_localstate[1]));
    }
    SECTION("Apply using dispatcher at a global wire") {
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

        sv.applyOperation("RX", {0}, false, {angles[0]});
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[0].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[0] == Pennylane::approx(expected_localstate[0]));
    }

    SECTION("Apply using dispatcher at a local wire") {
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

        sv.applyOperation("RX", {num_qubits - 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[1].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[1] == Pennylane::approx(expected_localstate[1]));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRY", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

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
    int subSvLength = 1 << nLocalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

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

    std::vector<std::vector<cp_t>> init_localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    std::vector<std::vector<cp_t>> expected_localstate(
        2, std::vector<cp_t>(subSvLength, {0.0, 0.0}));
    std::vector<std::vector<cp_t>> localstate(
        2, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    MPI_Scatter(initstate[0].data(), init_localstate[0].size(), message_type,
                init_localstate[0].data(), init_localstate[0].size(),
                message_type, 0, MPI_COMM_WORLD);
    MPI_Scatter(expected_results[0].data(), expected_localstate[0].size(),
                message_type, expected_localstate[0].data(),
                expected_localstate[0].size(), message_type, 0, MPI_COMM_WORLD);
    MPI_Scatter(expected_results[1].data(), expected_localstate[1].size(),
                message_type, expected_localstate[1].data(),
                expected_localstate[1].size(), message_type, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    SECTION("Apply directly at a global wire") {
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

        sv.applyRY({0}, false, angles[0]);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[0].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[0] == Pennylane::approx(expected_localstate[0]));
    }

    SECTION("Apply directly at a local wire") {
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

        sv.applyRY({num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[1].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[1] == Pennylane::approx(expected_localstate[1]));
    }
    SECTION("Apply using dispatcher at a global wire") {
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

        sv.applyOperation("RY", {0}, false, {angles[0]});
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[0].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[0] == Pennylane::approx(expected_localstate[0]));
    }

    SECTION("Apply using dispatcher at a local wire") {
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

        sv.applyOperation("RY", {num_qubits - 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[1].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[1] == Pennylane::approx(expected_localstate[1]));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::applyRZ", "[StateVectorCudaMPI_Param]",
                   float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    const size_t num_qubits = 4;

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
    int subSvLength = 1 << nLocalIndexBits;
    MPI_Barrier(MPI_COMM_WORLD);

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

    std::vector<std::vector<cp_t>> init_localstate(
        1, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    std::vector<std::vector<cp_t>> expected_localstate(
        2, std::vector<cp_t>(subSvLength, {0.0, 0.0}));
    std::vector<std::vector<cp_t>> localstate(
        2, std::vector<cp_t>(subSvLength, {0.0, 0.0}));

    MPI_Scatter(initstate[0].data(), init_localstate[0].size(), message_type,
                init_localstate[0].data(), init_localstate[0].size(),
                message_type, 0, MPI_COMM_WORLD);
    MPI_Scatter(expected_results[0].data(), expected_localstate[0].size(),
                message_type, expected_localstate[0].data(),
                expected_localstate[0].size(), message_type, 0, MPI_COMM_WORLD);
    MPI_Scatter(expected_results[1].data(), expected_localstate[1].size(),
                message_type, expected_localstate[1].data(),
                expected_localstate[1].size(), message_type, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    SECTION("Apply directly at a global wire") {
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

        sv.applyRZ({0}, false, angles[0]);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[0].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[0] == Pennylane::approx(expected_localstate[0]));
    }

    SECTION("Apply directly at a local wire") {
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

        sv.applyRZ({num_qubits - 1}, false, angles[0]);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[1].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[1] == Pennylane::approx(expected_localstate[1]));
    }
    SECTION("Apply using dispatcher at a global wire") {
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

        sv.applyOperation("RZ", {0}, false, {angles[0]});
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[0].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[0] == Pennylane::approx(expected_localstate[0]));
    }

    SECTION("Apply using dispatcher at a local wire") {
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

        sv.applyOperation("RZ", {num_qubits - 1}, false, {angles[0]});
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        sv.CopyGpuDataToHost(localstate[1].data(),
                             static_cast<std::size_t>(subSvLength));
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(localstate[1] == Pennylane::approx(expected_localstate[1]));
    }
}
