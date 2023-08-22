#pragma once

#include <algorithm>
#include <bit>
#include <complex>
#include <map>
#include <vector>

#include "MPIManager.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::CUDA;
} // namespace
/// @endcond
namespace Pennylane::MPI {

inline size_t reverseBits(size_t num, size_t nbits) {
    size_t reversed = 0;
    for (size_t i = 0; i < nbits; ++i) {
        // ith bit value
        size_t bit = (num & (1 << i)) >> i;
        reversed += bit << (nbits - i - 1);
    }
    return reversed;
}

template <class index_type>
inline std::tuple<std::vector<index_type>, std::vector<index_type>>
rankVector(const std::vector<index_type> &input) {
    // Create a copy of the input vector for sorting
    std::vector<index_type> sortedInput = input;

    // Sort the copy in ascending order
    std::sort(sortedInput.begin(), sortedInput.end());

    std::vector<index_type> ranks(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        auto it =
            std::lower_bound(sortedInput.begin(), sortedInput.end(), input[i]);
        ranks[i] = std::distance(sortedInput.begin(), it);
    }

    return {sortedInput, ranks};
}

/**
 * @brief Manage memory of Compressed Sparse Row (CSR) sparse matrix. CSR format
 * represents a matrix M by three (one-dimensional) arrays, that respectively
 * contain nonzero values, row offsets, and column indices.
 *
 * @tparam Precision Floating-point precision type.
 * @tparam index_type Integer type.
 */
template <class Precision, class index_type> class CSRMatrix {
  private:
    std::vector<index_type> columns_;
    std::vector<index_type> csrOffsets_;
    std::vector<std::complex<Precision>> values_;

  public:
    CSRMatrix(size_t num_rows, size_t nnz)
        : columns_(nnz, 0), csrOffsets_(num_rows + 1, 0), values_(nnz){};

    CSRMatrix(size_t num_rows, size_t nnz, const index_type *column_ptr,
              const index_type *csrOffsets_ptr, const std::complex<Precision> *value_ptr)
        : columns_(column_ptr, column_ptr + nnz),
          csrOffsets_(csrOffsets_ptr, csrOffsets_ptr + num_rows + 1),
          values_(value_ptr, value_ptr + nnz){};

    CSRMatrix() = default;

    /**
     * @brief Get the CSR format index vector of the matrix.
     */
    auto getColumns() -> std::vector<index_type> & { return columns_; }

    /**
     * @brief Get CSR format offset vector of the matrix.
     */
    auto getCsrOffsets() -> std::vector<index_type> & { return csrOffsets_; }

    /**
     * @brief Get CSR format data vector of the matrix.
     */
    auto getValues() -> std::vector<std::complex<Precision>> & {
        return values_;
    }

    auto matrixReorder() -> CSRMatrix<Precision, index_type>  {
        size_t num_rows = this->getCsrOffsets().size() - 1;
        size_t nnz = this->getColumns().size();
        size_t nbits = std::bit_width(num_rows) - 1;
        CSRMatrix<Precision, index_type> reorderedMatrix(num_rows, nnz);

        for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
            size_t org_row_idx = reverseBits(row_idx, nbits);

            size_t org_offset = this->getCsrOffsets()[org_row_idx];
            size_t local_col_size =
                this->getCsrOffsets()[org_row_idx + 1] - org_offset;

            reorderedMatrix.getCsrOffsets()[row_idx + 1] =
                local_col_size + reorderedMatrix.getCsrOffsets()[row_idx];

            // get unsorted column indices
            std::vector<index_type> local_col_indices(
                this->getColumns().begin() + org_offset,
                this->getColumns().begin() + org_offset + local_col_size);
            for (size_t i = 0; i < local_col_indices.size(); i++) {
                size_t col_idx = reverseBits(local_col_indices[i], nbits);
                local_col_indices[i] = col_idx;
            }

            auto [sorted_col_indices, ranks] =
                rankVector<index_type>(local_col_indices);

            for (size_t i = 0; i < sorted_col_indices.size(); i++) {
                reorderedMatrix
                    .getColumns()[reorderedMatrix.getCsrOffsets()[row_idx] +
                                  i] = sorted_col_indices[i];
                reorderedMatrix
                    .getValues()[reorderedMatrix.getCsrOffsets()[row_idx] + i] =
                    this->getValues()[org_offset + ranks[i]];

            }
        }
        return reorderedMatrix;
    }
};
} // namespace Pennylane::MPI