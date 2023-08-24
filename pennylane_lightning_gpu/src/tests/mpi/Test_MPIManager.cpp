#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "CSRMatrix.hpp"
#include "MPIManager.hpp"

#include "../TestHelpersLGPU.hpp"

using namespace Pennylane;
using namespace Pennylane::MPI;
using namespace CUDA;

TEMPLATE_TEST_CASE("MPIManager::Scatter", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply scatter") {
        std::vector<cp_t> sendBuf(size);
        int root = 0;
        cp_t result(2.0 * rank, 2.0 * rank + 1);
        if (rank == root) {
            for (size_t i = 0; i < sendBuf.size(); i++) {
                cp_t data(2.0 * i, 2.0 * i + 1);
                sendBuf[i] = data;
            }
        }

        auto recvBuf = mpi_manager.scatter<cp_t>(sendBuf, root);
        CHECK(recvBuf[0].real() == result.real());
        CHECK(recvBuf[0].imag() == result.imag());
    }

    SECTION("Apply Scatter") {
        std::vector<cp_t> sendBuf(size);
        std::vector<cp_t> recvBuf(1);
        int root = 0;
        cp_t result(2.0 * rank, 2.0 * rank + 1);
        if (rank == root) {
            for (size_t i = 0; i < sendBuf.size(); i++) {
                cp_t data(2.0 * i, 2.0 * i + 1);
                sendBuf[i] = data;
            }
        }

        mpi_manager.Scatter<cp_t>(sendBuf, recvBuf, root);
        CHECK(recvBuf[0].real() == result.real());
        CHECK(recvBuf[0].imag() == result.imag());
    }
}

TEMPLATE_TEST_CASE("MPIManager::Allgather", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply Allgather scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        std::vector<cp_t> recvBuf(size);

        mpi_manager.Allgather<cp_t>(sendBuf, recvBuf);

        for (size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply Allgather vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        std::vector<cp_t> recvBuf(mpi_manager.getSize());

        mpi_manager.Allgather<cp_t>(sendBuf, recvBuf);

        for (size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply allgather scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};

        auto recvBuf = mpi_manager.allgather<cp_t>(sendBuf);
        for (size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply allgather vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        auto recvBuf = mpi_manager.allgather<cp_t>(sendBuf);

        for (size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }
}

TEMPLATE_TEST_CASE("MPIManager::Allreduce", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply Allreduce scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        cp_t recvBuf;

        mpi_manager.Allreduce<cp_t>(sendBuf, recvBuf, "sum");
        CHECK(recvBuf.real() == static_cast<PrecisionT>((size - 1) * size / 2));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply allreduce scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        auto recvBuf = mpi_manager.allreduce<cp_t>(sendBuf, "sum");

        CHECK(recvBuf.real() == static_cast<PrecisionT>((size - 1) * size / 2));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Allreduce vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        std::vector<cp_t> recvBuf(1);

        mpi_manager.Allreduce<cp_t>(sendBuf, recvBuf, "sum");

        CHECK(recvBuf[0].real() ==
              static_cast<PrecisionT>((size - 1) * size / 2));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply allreduce vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        auto recvBuf = mpi_manager.allreduce<cp_t>(sendBuf, "sum");

        CHECK(recvBuf[0].real() ==
              static_cast<PrecisionT>((size - 1) * size / 2));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEMPLATE_TEST_CASE("MPIManager::Bcast", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    int rank = mpi_manager.getRank();

    SECTION("Apply Bcast scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        mpi_manager.Bcast<cp_t>(sendBuf, 0);
        CHECK(sendBuf.real() == static_cast<PrecisionT>(0));
        CHECK(sendBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Bcast vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        mpi_manager.Bcast<cp_t>(sendBuf, 0);
        CHECK(sendBuf[0].real() == static_cast<PrecisionT>(0));
        CHECK(sendBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEMPLATE_TEST_CASE("MPIManager::Sendrecv", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    SECTION("Apply Sendrecv scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0.0};
        cp_t recvBuf = {-1.0, -1.0};

        mpi_manager.Sendrecv<cp_t>(sendBuf, dest, recvBuf, source);

        CHECK(recvBuf.real() == static_cast<PrecisionT>(source));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Sendrecv vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0.0});
        std::vector<cp_t> recvBuf(1, {-1.0, -1.0});
        mpi_manager.Sendrecv<cp_t>(sendBuf, dest, recvBuf, source);
        CHECK(recvBuf[0].real() == static_cast<PrecisionT>(source));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEST_CASE("MPIManager::split") {
    MPIManager mpi_manager(MPI_COMM_WORLD);
    int rank = mpi_manager.getRank();
    int color = rank % 2;
    int key = rank;
    auto newComm = mpi_manager.split(color, key);
    CHECK(newComm.getSize() * 2 == mpi_manager.getSize());
}

TEMPLATE_TEST_CASE("CRSMatrix::Split", "[CRSMatrix]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    using index_type =
        typename std::conditional<std::is_same<TestType, float>::value, int32_t,
                                  int64_t>::type;

    MPIManager mpi_manager(MPI_COMM_WORLD);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    index_type csrOffsets[9] = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    index_type columns[16] = {0, 3, 1, 2, 1, 2, 0, 3, 4, 7, 5, 6, 5, 6, 4, 7};

    cp_t values[16] = {{1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
                       {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0},
                       {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
                       {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0}};

    index_type num_csrOffsets = 9;
    index_type num_rows = num_csrOffsets - 1;

    SECTION("Apply split") {
        if (rank == 0) {
            auto CSRMatVector = splitCSRMatrix<TestType, index_type>(
                mpi_manager, num_rows, csrOffsets, columns, values);

            std::vector<index_type> localcsrOffsets = {0, 2, 4, 6, 8};
            std::vector<index_type> local_indices = {0, 3, 1, 2, 1, 2, 0, 3};

            for (size_t i = 0; i < localcsrOffsets.size(); i++) {
                CHECK(CSRMatVector[0][0].getCsrOffsets()[i] ==
                      localcsrOffsets[i]);
                CHECK(CSRMatVector[1][1].getCsrOffsets()[i] ==
                      localcsrOffsets[i]);
            }

            for (size_t i = 0; i < local_indices.size(); i++) {
                CHECK(CSRMatVector[0][0].getColumns()[i] == local_indices[i]);
                CHECK(CSRMatVector[1][1].getColumns()[i] == local_indices[i]);
            }

            for (size_t i = 0; i < 8; i++) {
                CHECK(CSRMatVector[0][0].getValues()[i] == values[i]);
                CHECK(CSRMatVector[1][1].getValues()[i] == values[i + 8]);
            }

            CHECK(CSRMatVector[0][1].getValues().size() == 0);
            CHECK(CSRMatVector[1][0].getValues().size() == 0);
        }
    }

    SECTION("Apply SparseMatrix scatter") {
        std::vector<std::vector<CSRMatrix<TestType, index_type>>>
            csrmatrix_blocks;

        if (rank == 0) {
            csrmatrix_blocks = splitCSRMatrix<TestType, index_type>(
                mpi_manager, num_rows, csrOffsets, columns, values);
        }

        size_t local_num_rows = num_rows / size;

        std::vector<CSRMatrix<TestType, index_type>> localCSRMatVector;
        for (size_t i = 0; i < mpi_manager.getSize(); i++) {
            auto localCSRMat = scatterCSRMatrix<TestType, index_type>(
                mpi_manager, csrmatrix_blocks[i], local_num_rows, 0);
            localCSRMatVector.push_back(localCSRMat);
        }

        std::vector<index_type> localcsrOffsets = {0, 2, 4, 6, 8};
        std::vector<index_type> local_indices = {0, 3, 1, 2, 1, 2, 0, 3};

        if (rank == 0) {
            for (size_t i = 0; i < localcsrOffsets.size(); i++) {
                CHECK(localCSRMatVector[0].getCsrOffsets()[i] ==
                      localcsrOffsets[i]);
            }

            for (size_t i = 0; i < local_indices.size(); i++) {
                CHECK(localCSRMatVector[0].getColumns()[i] == local_indices[i]);
            }

            for (size_t i = 0; i < 8; i++) {
                CHECK(localCSRMatVector[0].getValues()[i] == values[i]);
            }

            CHECK(localCSRMatVector[1].getValues().size() == 0);
        } else {
            for (size_t i = 0; i < localcsrOffsets.size(); i++) {
                CHECK(localCSRMatVector[1].getCsrOffsets()[i] ==
                      localcsrOffsets[i]);
            }

            for (size_t i = 0; i < local_indices.size(); i++) {
                CHECK(localCSRMatVector[1].getColumns()[i] == local_indices[i]);
            }

            for (size_t i = 0; i < 8; i++) {
                CHECK(localCSRMatVector[1].getValues()[i] == values[i + 8]);
            }

            CHECK(localCSRMatVector[0].getValues().size() == 0);
        }
    }
}