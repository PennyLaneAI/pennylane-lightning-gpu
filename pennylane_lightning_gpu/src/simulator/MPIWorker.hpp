#pragma once
#include <algorithm>
#include <bit>
#include <string>
#include <vector>

#include "MPIManager.hpp"
#include "cuda_helpers.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <custatevec.h>
/// @cond DEV
namespace {
namespace cuUtil = Pennylane::CUDA::Util;
using namespace Pennylane::CUDA;
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::MPI {

enum WireStatus { Default, Target, Control };

/**
 * @brief Create wire pairs for bit index swap and transform all control and
 * target wires to local ones.
 *
 * @param numLocalQubits Number of local qubits.
 * @param numTotalQubits Number of total qubits.
 * @param ctrls Vector of control wires.
 * @param tgts Vector of target wires.
 * @return wirePairs Wire pairs to be passed to SV bit index swap worker.
 */
inline std::vector<int2> createWirePairs(const int numLocalQubits,
                                         const int numTotalQubits,
                                         std::vector<int> &ctrls,
                                         std::vector<int> &tgts,
                                         std::vector<int> &statusWires) {
    std::vector<int2> wirePairs;
    int localbit = numLocalQubits - 1, globalbit = numLocalQubits;
    while (localbit >= 0 && globalbit < numTotalQubits) {
        if (statusWires[localbit] == 0 && statusWires[globalbit] != 0) {
            int2 wirepair = make_int2(localbit, globalbit);
            wirePairs.push_back(wirepair);
            if (statusWires[globalbit] == WireStatus::Control) {
                for (size_t k = 0; k < ctrls.size(); k++) {
                    if (ctrls[k] == globalbit) {
                        ctrls[k] = localbit;
                    }
                }
            } else {
                for (size_t k = 0; k < tgts.size(); k++) {
                    if (tgts[k] == globalbit) {
                        tgts[k] = localbit;
                    }
                }
            }
            std::swap(statusWires[localbit], statusWires[globalbit]);
        } else {
            if (statusWires[localbit] != WireStatus::Default) {
                localbit--;
            }
            if (statusWires[globalbit] == WireStatus::Default) {
                globalbit++;
            }
        }
    }
    return wirePairs;
}

/**
 * @brief Utility function object to tell std::shared_ptr how to
 * release/destroy various custatevecSVSwapWorker related objects.
 */
struct MPIWorkerDeleter {
    void operator()(cudaStream_t localStream) const {
        PL_CUDA_IS_SUCCESS(cudaStreamDestroy(localStream));
    }

    void operator()(custatevecSVSwapWorkerDescriptor_t svSegSwapWorker,
                    custatevecHandle_t handle,
                    custatevecCommunicatorDescriptor_t communicator,
                    void *d_extraWorkspace, void *d_transferWorkspace,
                    std::vector<void *> d_subSVsP2P,
                    std::vector<cudaEvent_t> remoteEvents,
                    cudaEvent_t localEvent) const {
        PL_CUSTATEVEC_IS_SUCCESS(
            custatevecSVSwapWorkerDestroy(handle, svSegSwapWorker));
        PL_CUSTATEVEC_IS_SUCCESS(
            custatevecCommunicatorDestroy(handle, communicator));
        PL_CUDA_IS_SUCCESS(cudaFree(d_extraWorkspace));
        PL_CUDA_IS_SUCCESS(cudaFree(d_transferWorkspace));
        for (auto *d_subSV : d_subSVsP2P)
            PL_CUDA_IS_SUCCESS(cudaIpcCloseMemHandle(d_subSV));
        for (auto event : remoteEvents)
            PL_CUDA_IS_SUCCESS(cudaEventDestroy(event));
        PL_CUDA_IS_SUCCESS(cudaEventDestroy(localEvent));
    }
};

using SharedLocalStream =
    std::shared_ptr<std::remove_pointer<cudaStream_t>::type>;
using SharedMPIWorker = std::shared_ptr<
    std::remove_pointer<custatevecSVSwapWorkerDescriptor_t>::type>;

/**
 * @brief Creates a SharedLocalStream (a shared pointer to a cuda stream)
 */
inline SharedLocalStream make_shared_local_stream() {
    cudaStream_t localStream;
    PL_CUDA_IS_SUCCESS(cudaStreamCreate(&localStream));
    return {localStream, MPIWorkerDeleter()};
}

/**
 * @brief Creates a SharedMPIWorker (a shared pointer to a
 * custatevecSVSwapWorker)
 *
 * @param handle custatevecHandle.
 * @param mpi_manager MPI manager object.
 * @param mpi_buffer_size Size to set MPI buffer.
 * @param sv Pointer to the data requires MPI operation.
 * @param numLocalQubits Number of local qubits.
 * @param localStream Local cuda stream.
 */

template <typename CFP_t>
inline SharedMPIWorker
make_shared_mpi_worker(custatevecHandle_t handle, MPIManager &mpi_manager,
                       const size_t mpi_buffer_size, CFP_t *sv,
                       const size_t numLocalQubits, cudaStream_t localStream) {

    custatevecSVSwapWorkerDescriptor_t svSegSwapWorker = nullptr;

    int nDevices_int = 0;
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices_int));

    size_t nDevices = static_cast<size_t>(nDevices_int);

    // Ensure the number of P2P devices is calculated based on the number of MPI
    // processes within the node
    nDevices = mpi_manager.getSizeNode() < nDevices ? mpi_manager.getSizeNode()
                                                    : nDevices;

    size_t nP2PDeviceBits = std::bit_width(nDevices) - 1;

    cudaDataType_t svDataType;
    if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                  std::is_same_v<CFP_t, double2>) {
        svDataType = CUDA_C_64F;
    } else {
        svDataType = CUDA_C_32F;
    }

    cudaEvent_t localEvent = nullptr;
    custatevecCommunicatorDescriptor_t communicator = nullptr;

    PL_CUDA_IS_SUCCESS(cudaEventCreateWithFlags(
        &localEvent, cudaEventInterprocess | cudaEventDisableTiming));

    custatevecCommunicatorType_t communicatorType;
    if (mpi_manager.getVendor() == "MPICH") {
        communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_MPICH;
    }
    if (mpi_manager.getVendor() == "Open MPI") {
        communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
    }

    auto err = custatevecCommunicatorCreate(handle, &communicator,
                                            communicatorType, nullptr);
    if (err != CUSTATEVEC_STATUS_SUCCESS) {
        communicator = nullptr;
        PL_CUSTATEVEC_IS_SUCCESS(custatevecCommunicatorCreate(
            handle, &communicator, communicatorType, "libmpi.so"));
    }
    mpi_manager.Barrier();

    void *d_extraWorkspace = nullptr;
    void *d_transferWorkspace = nullptr;

    std::vector<void *> d_subSVsP2P;
    std::vector<int> subSVIndicesP2P;
    std::vector<cudaEvent_t> remoteEvents;

    size_t extraWorkspaceSize = 0;
    size_t minTransferWorkspaceSize = 0;

    PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerCreate(
        /* custatevecHandle_t */ handle,
        /* custatevecSVSwapWorkerDescriptor_t* */ &svSegSwapWorker,
        /* custatevecCommunicatorDescriptor_t */ communicator,
        /* void* */ sv,
        /* int32_t */ mpi_manager.getRank(),
        /* cudaEvent_t */ localEvent,
        /* cudaDataType_t */ svDataType,
        /* cudaStream_t */ localStream,
        /* size_t* */ &extraWorkspaceSize,
        /* size_t* */ &minTransferWorkspaceSize));

    PL_CUDA_IS_SUCCESS(cudaMalloc(&d_extraWorkspace, extraWorkspaceSize));

    PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerSetExtraWorkspace(
        /* custatevecHandle_t */ handle,
        /* custatevecSVSwapWorkerDescriptor_t */ svSegSwapWorker,
        /* void* */ d_extraWorkspace,
        /* size_t */ extraWorkspaceSize));

    size_t transferWorkspaceSize;
    if (mpi_buffer_size == 0) {
        // Here 26 is based on the benchmark tests on the Perlmutter.
        transferWorkspaceSize =
            size_t{1} << (numLocalQubits < 26 ? (numLocalQubits) : 26);
    } else {
        transferWorkspaceSize = size_t{1} << mpi_buffer_size;
        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            transferWorkspaceSize = transferWorkspaceSize * sizeof(double) * 2;
        } else {
            transferWorkspaceSize = transferWorkspaceSize * sizeof(float) * 2;
        }
    }

    transferWorkspaceSize =
        std::max(minTransferWorkspaceSize, transferWorkspaceSize);
    PL_CUDA_IS_SUCCESS(cudaMalloc(&d_transferWorkspace, transferWorkspaceSize));

    PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerSetTransferWorkspace(
        /* custatevecHandle_t */ handle,
        /* custatevecSVSwapWorkerDescriptor_t */ svSegSwapWorker,
        /* void* */ d_transferWorkspace,
        /* size_t */ transferWorkspaceSize));

    if (nP2PDeviceBits != 0) {
        cudaIpcMemHandle_t ipcMemHandle;
        PL_CUDA_IS_SUCCESS(cudaIpcGetMemHandle(&ipcMemHandle, sv));
        std::vector<cudaIpcMemHandle_t> ipcMemHandles(mpi_manager.getSize());

        mpi_manager.Allgather<cudaIpcMemHandle_t>(ipcMemHandle, ipcMemHandles,
                                                  sizeof(ipcMemHandle));
        cudaIpcEventHandle_t eventHandle;
        PL_CUDA_IS_SUCCESS(cudaIpcGetEventHandle(&eventHandle, localEvent));
        // distribute event handles
        std::vector<cudaIpcEventHandle_t> ipcEventHandles(
            mpi_manager.getSize());

        mpi_manager.Allgather<cudaIpcEventHandle_t>(
            eventHandle, ipcEventHandles, sizeof(eventHandle));
        //  get remove device pointers and events
        size_t nSubSVsP2P = size_t{1} << nP2PDeviceBits;
        size_t p2pSubSVIndexBegin =
            (mpi_manager.getRank() / nSubSVsP2P) * nSubSVsP2P;
        size_t p2pSubSVIndexEnd = p2pSubSVIndexBegin + nSubSVsP2P;
        for (size_t p2pSubSVIndex = p2pSubSVIndexBegin;
             p2pSubSVIndex < p2pSubSVIndexEnd; p2pSubSVIndex++) {
            if (static_cast<size_t>(mpi_manager.getRank()) == p2pSubSVIndex)
                continue; // don't need local sub state vector pointer
            void *d_subSVP2P = nullptr;
            const auto &dstMemHandle = ipcMemHandles[p2pSubSVIndex];
            PL_CUDA_IS_SUCCESS(cudaIpcOpenMemHandle(
                &d_subSVP2P, dstMemHandle, cudaIpcMemLazyEnablePeerAccess));
            d_subSVsP2P.push_back(d_subSVP2P);
            cudaEvent_t eventP2P = nullptr;
            PL_CUDA_IS_SUCCESS(cudaIpcOpenEventHandle(
                &eventP2P, ipcEventHandles[p2pSubSVIndex]));
            remoteEvents.push_back(eventP2P);
            subSVIndicesP2P.push_back(p2pSubSVIndex);
        }

        // set p2p sub state vectors
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerSetSubSVsP2P(
            /* custatevecHandle_t */ handle,
            /* custatevecSVSwapWorkerDescriptor_t */ svSegSwapWorker,
            /* void** */ d_subSVsP2P.data(),
            /* const int32_t* */ subSVIndicesP2P.data(),
            /* cudaEvent_t */ remoteEvents.data(),
            /* const uint32_t */ static_cast<uint32_t>(d_subSVsP2P.size())));
    }

    return {svSegSwapWorker,
            std::bind(MPIWorkerDeleter(), std::placeholders::_1, handle,
                      communicator, d_extraWorkspace, d_transferWorkspace,
                      d_subSVsP2P, remoteEvents, localEvent)};
}
} // namespace Pennylane::MPI