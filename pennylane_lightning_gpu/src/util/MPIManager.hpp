#pragma once

#include <complex>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "Error.hpp"

namespace Pennylane::MPI {
// check if num is a power of 2
inline bool isPow2(int num) {
    PL_ABORT_IF_NOT(num > 0, "Operand of isPow2 should be positive.");
    return (num & (num - 1)) == 0;
}

inline void errhandler(int errcode, const char *str) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    fprintf(stderr, "%s: %s\n", str, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define PL_MPI_IS_SUCCESS(fn)                                                  \
    {                                                                          \
        int errcode;                                                           \
        errcode = (fn);                                                        \
        if (errcode != MPI_SUCCESS)                                            \
            errhandler(errcode, #fn);                                          \
    }

#define CPPTYPE(s) std::type_index(typeid(s))

class MPIManager {
  private:
    bool isExternalComm_;
    int rank_;
    int size_per_node_;
    int size_;
    MPI_Comm communiator_;

    std::string vendor_;
    int version_;
    int subversion_;

  public:
  
    MPIManager() : communiator_(MPI_COMM_WORLD) {
        isExternalComm_ = true;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communiator_, &rank_));
        PL_MPI_IS_SUCCESS(MPI_Comm_size(communiator_, &size_));
        findVendor();
        findVersion();
        getNumProcsPerNode();
    }

    MPIManager(MPI_Comm communicator) : communiator_(communicator) {
        isExternalComm_ = true;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communiator_, &rank_));
        PL_MPI_IS_SUCCESS(MPI_Comm_size(communiator_, &size_));
        findVendor();
        findVersion();
        getNumProcsPerNode();
    }

    MPIManager(int argc, char **argv) {
        PL_MPI_IS_SUCCESS(MPI_Init(&argc, &argv));
        isExternalComm_ = false;
        communiator_ = MPI_COMM_WORLD;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communiator_, &rank_));
        PL_MPI_IS_SUCCESS(MPI_Comm_size(communiator_, &size_));
        findVendor();
        findVersion();
        check_mpi_config();
        getNumProcsPerNode();
    }

    MPIManager(MPIManager &other) {
        isExternalComm_ = true;
        rank_ = other.rank_;
        size_ = other.size_;
        communiator_ = other.communiator_;
        vendor_ = other.vendor_;
        version_ = other.version_;
        subversion_ = other.subversion_;
        size_per_node_ = other.size_per_node_;
    }

    ~MPIManager() {
        if (!isExternalComm_) {
            int initflag;
            int finflag;
            PL_MPI_IS_SUCCESS(MPI_Initialized(&initflag));
            PL_MPI_IS_SUCCESS(MPI_Finalized(&finflag));
            if (initflag && !finflag) {
                PL_MPI_IS_SUCCESS(MPI_Finalize());
            }
        }
    }

    // General MPI operations
    auto getRank() const { return rank_; }

    auto getSize() const { return size_; }

    auto getSizeNode() const { return size_per_node_; }

    auto getComm() const { return communiator_; }

    auto getTime() const { return MPI_Wtime(); }

    auto getVendor() const { return vendor_; }

    auto getVersion() -> std::tuple<int, int> {
        return {version_, subversion_};
    }

    void findVendor() {
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        int resultlen;

        PL_MPI_IS_SUCCESS(MPI_Get_library_version(version, &resultlen));

        if (strstr(version, "Open MPI") != nullptr) {
            vendor_ = "Open MPI";
        } else if (strstr(version, "MPICH") != NULL) {
            vendor_ = "MPICH";
        } else {
            std::cerr << "Unsupported MPI implementation.\n";
            exit(EXIT_FAILURE);
        }
    }

    void findVersion() {
        PL_MPI_IS_SUCCESS(MPI_Get_version(&version_, &subversion_));
    }

    void getNumProcsPerNode() {
        MPI_Comm node_comm;
        PL_MPI_IS_SUCCESS(
            MPI_Comm_split_type(this->getComm(), MPI_COMM_TYPE_SHARED,
                                this->getRank(), MPI_INFO_NULL, &node_comm));
        PL_MPI_IS_SUCCESS(MPI_Comm_size(node_comm, &size_per_node_));
        this->Barrier();
        PL_ABORT_IF(isPow2(size_per_node_) != true,
                    "Number of processes per node is not power of two.");
    }

    void check_mpi_config() {
        // check if number of processes is power of two.
        // This is required by custatevec
        PL_ABORT_IF(isPow2(this->getSize()) != true,
                    "Processes number is not power of two.");
        // Check if number of processes per node is power of two.
        // This is required by custatevec
        PL_ABORT_IF(isPow2(size_per_node_) != true,
                    "Processes number per node is not power of two.");
    }

    template <typename T>
    void Allgather(T &sendBuf, std::vector<T> &recvBuf, int sendCount = 1) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_ABORT_IF(recvBuf.size() != static_cast<size_t>(this->getSize()),
                    "Incompatible size of sendBuf and recvBuf.");
        PL_MPI_IS_SUCCESS(MPI_Allgather(&sendBuf, sendCount, datatype,
                                        recvBuf.data(), sendCount, datatype,
                                        this->getComm()));
    }

    template <typename T>
    auto allgather(T &sendBuf, int sendCount = 1) -> std::vector<T> {
        MPI_Datatype datatype = getMPIDatatype<T>();
        std::vector<T> recvBuf(this->getSize() * sendCount);
        PL_MPI_IS_SUCCESS(MPI_Allgather(&sendBuf, sendCount, datatype,
                                        recvBuf.data(), sendCount, datatype,
                                        this->getComm()));
        return recvBuf;
    }

    template <typename T>
    void Allgather(std::vector<T> &sendBuf, std::vector<T> &recvBuf) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_ABORT_IF(recvBuf.size() != sendBuf.size() * this->getSize(),
                    "Incompatible size of sendBuf and recvBuf.");
        PL_MPI_IS_SUCCESS(MPI_Allgather(
            sendBuf.data(), sendBuf.size(), datatype, recvBuf.data(),
            sendBuf.size(), datatype, this->getComm()));
    }

    template <typename T>
    auto allgather(std::vector<T> &sendBuf) -> std::vector<T> {
        MPI_Datatype datatype = getMPIDatatype<T>();
        std::vector<T> recvBuf(sendBuf.size() * this->getSize());
        PL_MPI_IS_SUCCESS(MPI_Allgather(
            sendBuf.data(), sendBuf.size(), datatype, recvBuf.data(),
            sendBuf.size(), datatype, this->getComm()));
        return recvBuf;
    }

    template <typename T>
    void Allreduce(T &sendBuf, T &recvBuf, const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Allreduce(&sendBuf, &recvBuf, 1, datatype, op,
                                        this->getComm()));
    }

    template <typename T>
    auto allreduce(T &sendBuf, const std::string &op_str) -> T {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        T recvBuf;
        PL_MPI_IS_SUCCESS(MPI_Allreduce(&sendBuf, &recvBuf, 1, datatype, op,
                                        this->getComm()));
        return recvBuf;
    }

    template <typename T>
    void Allreduce(std::vector<T> &sendBuf, std::vector<T> &recvBuf,
                   const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Allreduce(sendBuf.data(), recvBuf.data(),
                                        sendBuf.size(), datatype, op,
                                        this->getComm()));
    }

    template <typename T>
    auto allreduce(std::vector<T> &sendBuf, const std::string &op_str)
        -> std::vector<T> {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        std::vector<T> recvBuf(sendBuf.size());
        PL_MPI_IS_SUCCESS(MPI_Allreduce(sendBuf.data(), recvBuf.data(), 1,
                                        datatype, op, this->getComm()));
        return recvBuf;
    }

    void Barrier() { PL_MPI_IS_SUCCESS(MPI_Barrier(this->getComm())); }

    template <typename T> void Bcast(T &sendBuf, int root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_MPI_IS_SUCCESS(
            MPI_Bcast(&sendBuf, 1, datatype, root, this->getComm()));
    }

    template <typename T> void Bcast(std::vector<T> &sendBuf, int root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_MPI_IS_SUCCESS(MPI_Bcast(sendBuf.data(), sendBuf.size(), datatype,
                                    root, this->getComm()));
    }

    template <typename T>
    void Scatter(std::vector<T> &sendBuf, std::vector<T> &recvBuf, int root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_ABORT_IF(sendBuf.size() != recvBuf.size() * this->getSize(),
                    "Incompatible size of sendBuf and recvBuf.");
        PL_MPI_IS_SUCCESS(MPI_Scatter(sendBuf.data(), recvBuf.size(), datatype,
                                      recvBuf.data(), recvBuf.size(), datatype,
                                      root, this->getComm()));
    }

    template <typename T>
    auto scatter(std::vector<T> &sendBuf, int root) -> std::vector<T> {
        MPI_Datatype datatype = getMPIDatatype<T>();
        int recvBufSize = sendBuf.size() / this->getSize();
        std::vector<T> recvBuf(recvBufSize);
        PL_MPI_IS_SUCCESS(MPI_Scatter(sendBuf.data(), recvBuf.size(), datatype,
                                      recvBuf.data(), recvBuf.size(), datatype,
                                      root, this->getComm()));
        return recvBuf;
    }

    template <typename T>
    void Sendrecv(T &sendBuf, int dest, T &recvBuf, int source) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Status status;
        int sendtag = 0;
        int recvtag = 0;
        PL_MPI_IS_SUCCESS(MPI_Sendrecv(&sendBuf, 1, datatype, dest, sendtag,
                                       &recvBuf, 1, datatype, source, recvtag,
                                       this->getComm(), &status));
    }

    template <typename T>
    void Sendrecv(std::vector<T> &sendBuf, int dest, std::vector<T> &recvBuf,
                  int source) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Status status;
        int sendtag = 0;
        int recvtag = 0;
        PL_MPI_IS_SUCCESS(MPI_Sendrecv(sendBuf.data(), sendBuf.size(), datatype,
                                       dest, sendtag, recvBuf.data(),
                                       recvBuf.size(), datatype, source,
                                       recvtag, this->getComm(), &status));
    }

    auto split(int color, int key) -> MPIManager {
        MPI_Comm newcomm;
        PL_MPI_IS_SUCCESS(
            MPI_Comm_split(this->getComm(), color, key, &newcomm));
        return MPIManager(newcomm);
    }

  private:
    template <typename T> auto getMPIDatatype() -> MPI_Datatype {
        auto it = cpp_mpi_type_map.find(CPPTYPE(T));
        if (it != cpp_mpi_type_map.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Type not supported");
        }
    }

    auto getMPIOpType(const std::string &op_str) -> MPI_Op {
        auto it = cpp_mpi_op_map.find(op_str);
        if (it != cpp_mpi_op_map.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Op not supported");
        }
    }

    std::unordered_map<std::string, MPI_Datatype> cpp_mpi_op_map = {
        {"op_null", MPI_OP_NULL}, {"max", MPI_MAX},
        {"min", MPI_MIN},         {"sum", MPI_SUM},
        {"prod", MPI_PROD},       {"land", MPI_LAND},
        {"band", MPI_BAND},       {"lor", MPI_LOR},
        {"bor", MPI_BOR},         {"lxor", MPI_LXOR},
        {"bxor", MPI_BXOR},       {"minloc", MPI_MINLOC},
        {"maxloc", MPI_MAXLOC},   {"replace", MPI_REPLACE},
    };

    std::unordered_map<std::type_index, MPI_Datatype> cpp_mpi_type_map = {
        {CPPTYPE(char), MPI_CHAR},
        {CPPTYPE(signed char), MPI_SIGNED_CHAR},
        {CPPTYPE(unsigned char), MPI_UNSIGNED_CHAR},
        {CPPTYPE(wchar_t), MPI_WCHAR},
        {CPPTYPE(short), MPI_SHORT},
        {CPPTYPE(unsigned short), MPI_UNSIGNED_SHORT},
        {CPPTYPE(int), MPI_INT},
        {CPPTYPE(unsigned int), MPI_UNSIGNED},
        {CPPTYPE(long), MPI_LONG},
        {CPPTYPE(unsigned long), MPI_UNSIGNED_LONG},
        {CPPTYPE(long long), MPI_LONG_LONG_INT},
        {CPPTYPE(float), MPI_FLOAT},
        {CPPTYPE(double), MPI_DOUBLE},
        {CPPTYPE(long double), MPI_LONG_DOUBLE},
        {CPPTYPE(int8_t), MPI_INT8_T},
        {CPPTYPE(int16_t), MPI_INT16_T},
        {CPPTYPE(int32_t), MPI_INT32_T},
        {CPPTYPE(int64_t), MPI_INT64_T},
        {CPPTYPE(uint8_t), MPI_UINT8_T},
        {CPPTYPE(uint16_t), MPI_UINT16_T},
        {CPPTYPE(uint32_t), MPI_UINT32_T},
        {CPPTYPE(uint64_t), MPI_UINT64_T},
        {CPPTYPE(bool), MPI_C_BOOL},
        {CPPTYPE(std::complex<float>), MPI_C_FLOAT_COMPLEX},
        {CPPTYPE(std::complex<double>), MPI_C_DOUBLE_COMPLEX},
        {CPPTYPE(std::complex<long double>), MPI_C_LONG_DOUBLE_COMPLEX},
        // cuda related types
        {CPPTYPE(cudaIpcMemHandle_t), MPI_INT8_T},
    };
};
} // namespace Pennylane::MPI