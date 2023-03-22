// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file
 * Defines a class for the measurement of observables
 * in quantum states represented by the StateVectorCuda classes.
 */

#pragma once
#include "StateVectorCudaBase.hpp"

namespace Pennylane::CUDA {

template <class fp_t = double, class SVType = StateVectorCudaManaged<fp_t>>
class Measurements {
  private:
    const SVType &sv_ref;
    using CFP_t = SVType::CFP_t;

    /**
     * @brief Get expectation of a given host-defined matrix.
     *
     * @param matrix Host-defined row-major order gate matrix.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto getExpectationValueHostMatrix(const std::vector<CFP_t> &matrix,
                                       const std::vector<std::size_t> &tgts) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        size_t nIndexBits = BaseType::getNumQubits();
        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        CFP_t expect;

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect,
            /* cudaDataType_t */ data_type,
            /* double* */ nullptr,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        return expect;
    }

    /**
     * @brief Get expectation of a given host or device defined array.
     *
     * @param matrix Host or device defined row-major order gate matrix array.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto getExpectationValueDeviceMatrix(const CFP_t *matrix,
                                         const std::vector<std::size_t> &tgts) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        size_t nIndexBits = BaseType::getNumQubits();
        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const uint32_t */ tgtsInt.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        CFP_t expect;

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect,
            /* cudaDataType_t */ data_type,
            /* double* */ nullptr,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgtsInt.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        return expect;
    }

    /**
     * @brief Get expectation value for a sum of Pauli words.
     *
     * @param pauli_words Vector of Pauli-words to evaluate expectation value.
     * @param tgts Coupled qubit index to apply each Pauli term.
     * @param coeffs Numpy array buffer of size |pauli_words|
     * @return auto Expectation value.
     */
    auto getExpectationValuePauliWords(
        const std::vector<std::string> &pauli_words,
        const std::vector<std::vector<std::size_t>> &tgts,
        const std::complex<Precision> *coeffs) {

        uint32_t nIndexBits = static_cast<uint32_t>(BaseType::getNumQubits());
        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        // Note: due to API design, cuStateVec assumes this is always a double.
        // Push NVIDIA to move this to behind API for future releases, and
        // support 32/64 bits.
        std::vector<double> expect(pauli_words.size());

        std::vector<std::vector<custatevecPauli_t>> pauliOps;

        std::vector<custatevecPauli_t *> pauliOps_ptr;

        for (auto &p_word : pauli_words) {
            pauliOps.push_back(cuUtil::pauliStringToEnum(p_word));
            pauliOps_ptr.push_back((*pauliOps.rbegin()).data());
        }

        std::vector<std::vector<int32_t>> basisBits;
        std::vector<int32_t *> basisBits_ptr;
        std::vector<uint32_t> n_basisBits;

        for (auto &wires : tgts) {
            std::vector<int32_t> wiresInt(wires.size());
            std::transform(wires.begin(), wires.end(), wiresInt.begin(),
                           [&](std::size_t x) {
                               return static_cast<int>(
                                   BaseType::getNumQubits() - 1 - x);
                           });
            basisBits.push_back(wiresInt);
            basisBits_ptr.push_back((*basisBits.rbegin()).data());
            n_basisBits.push_back(wiresInt.size());
        }

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationsOnPauliBasis(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double* */ expect.data(),
            /* const custatevecPauli_t ** */
            const_cast<const custatevecPauli_t **>(pauliOps_ptr.data()),
            /* const uint32_t */ static_cast<uint32_t>(pauliOps.size()),
            /* const int32_t ** */
            const_cast<const int32_t **>(basisBits_ptr.data()),
            /* const uint32_t */ n_basisBits.data()));

        std::complex<Precision> result{0, 0};

        if constexpr (std::is_same_v<Precision, double>) {
            for (std::size_t idx = 0; idx < expect.size(); idx++) {
                result += expect[idx] * coeffs[idx];
            }
            return std::real(result);
        } else {
            std::vector<Precision> expect_cast(expect.size());
            std::transform(expect.begin(), expect.end(), expect_cast.begin(),
                           [](double x) { return static_cast<float>(x); });

            for (std::size_t idx = 0; idx < expect_cast.size(); idx++) {
                result += expect_cast[idx] * expect_cast[idx];
            }

            return std::real(result);
        }
    }

    /**
     * @brief Utility method for samples.
     *
     * @param num_samples Number of Samples
     *
     * @return std::vector<size_t> A 1-d array storing the samples.
     * Each sample has a length equal to the number of qubits. Each sample can
     * be accessed using the stride sample_id*num_qubits, where sample_id is a
     * number between 0 and num_samples-1.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {

        std::vector<double> rand_nums(num_samples);
        custatevecSamplerDescriptor_t sampler;

        const size_t num_qubits = BaseType::getNumQubits();
        const int bitStringLen = BaseType::getNumQubits();

        std::vector<int> bitOrdering(num_qubits);
        std::iota(std::begin(bitOrdering), std::end(bitOrdering),
                  0); // Fill with 0, 1, ...,

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<Precision> dis(0.0, 1.0);
        for (size_t n = 0; n < num_samples; n++) {
            rand_nums[n] = dis(gen);
        }
        std::vector<size_t> samples(num_samples * num_qubits, 0);
        std::unordered_map<size_t, size_t> cache;
        std::vector<custatevecIndex_t> bitStrings(num_samples);

        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        // create sampler and check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerCreate(
            handle_.get(), BaseType::getData(), data_type, num_qubits, &sampler,
            num_samples, &extraWorkspaceSizeInBytes));

        // allocate external workspace if necessary
        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

        // sample preprocess
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerPreprocess(
            handle_.get(), sampler, extraWorkspace, extraWorkspaceSizeInBytes));

        // sample bit strings
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerSample(
            handle_.get(), sampler, bitStrings.data(), bitOrdering.data(),
            bitStringLen, rand_nums.data(), num_samples,
            CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

        // destroy descriptor and handle
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerDestroy(sampler));

        // Pick samples
        for (size_t i = 0; i < num_samples; i++) {
            auto idx = bitStrings[i];
            // If cached, retrieve sample from cache
            if (cache.count(idx) != 0) {
                size_t cache_id = cache[idx];
                auto it_temp = samples.begin() + cache_id * num_qubits;
                std::copy(it_temp, it_temp + num_qubits,
                          samples.begin() + i * num_qubits);
            }
            // If not cached, compute
            else {
                for (size_t j = 0; j < num_qubits; j++) {
                    samples[i * num_qubits + (num_qubits - 1 - j)] =
                        (idx >> j) & 1U;
                }
                cache[idx] = i;
            }
        }

        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));

        return samples;
    }

    /**
     * @brief expval(H) calculation with cuSparseSpMV.
     *
     * @tparam index_type Integer type used as indices of the sparse matrix.
     * @param csr_Offsets_ptr Pointer to the array of row offsets of the sparse
     * matrix. Array of size csrOffsets_size.
     * @param csrOffsets_size Number of Row offsets of the sparse matrix.
     * @param columns_ptr Pointer to the array of column indices of the sparse
     * matrix. Array of size numNNZ
     * @param values_ptr Pointer to the array of the non-zero elements
     * @param numNNZ Number of non-zero elements.
     * @return auto Expectation value.
     */
    template <class index_type>
    auto getExpectationValueOnSparseSpMV(
        const index_type *csrOffsets_ptr, const index_type csrOffsets_size,
        const index_type *columns_ptr,
        const std::complex<Precision> *values_ptr, const index_type numNNZ) {

        const index_type nIndexBits = BaseType::getNumQubits();
        const index_type length = 1 << nIndexBits;
        const int64_t num_rows = static_cast<int64_t>(
            csrOffsets_size -
            1); // int64_t is required for num_rows by cusparseCreateCsr
        const int64_t num_cols = static_cast<int64_t>(
            num_rows); // int64_t is required for num_cols by cusparseCreateCsr
        const int64_t nnz = static_cast<int64_t>(
            numNNZ); // int64_t is required for nnz by cusparseCreateCsr

        const CFP_t alpha = {1.0, 0.0};
        const CFP_t beta = {0.0, 0.0};

        Precision expect = 0;

        auto device_id = BaseType::getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = BaseType::getDataBuffer().getDevTag().getStreamID();

        DataBuffer<index_type, int> d_csrOffsets{
            static_cast<std::size_t>(csrOffsets_size), device_id, stream_id,
            true};
        DataBuffer<index_type, int> d_columns{static_cast<std::size_t>(numNNZ),
                                              device_id, stream_id, true};
        DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(numNNZ),
                                        device_id, stream_id, true};
        DataBuffer<CFP_t, int> d_tmp{static_cast<std::size_t>(length),
                                     device_id, stream_id, true};

        d_csrOffsets.CopyHostDataToGpu(csrOffsets_ptr, d_csrOffsets.getLength(),
                                       false);
        d_columns.CopyHostDataToGpu(columns_ptr, d_columns.getLength(), false);
        d_values.CopyHostDataToGpu(values_ptr, d_values.getLength(), false);

        cudaDataType_t data_type;
        cusparseIndexType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSPARSE_INDEX_64I;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSPARSE_INDEX_32I;
        }

        // CUSPARSE APIs
        cusparseSpMatDescr_t mat;
        cusparseDnVecDescr_t vecX, vecY;

        size_t bufferSize = 0;
        cusparseHandle_t handle = getCusparseHandle();

        // Create sparse matrix A in CSR format
        PL_CUSPARSE_IS_SUCCESS(cusparseCreateCsr(
            /* cusparseSpMatDescr_t* */ &mat,
            /* int64_t */ num_rows,
            /* int64_t */ num_cols,
            /* int64_t */ nnz,
            /* void* */ d_csrOffsets.getData(),
            /* void* */ d_columns.getData(),
            /* void* */ d_values.getData(),
            /* cusparseIndexType_t */ compute_type,
            /* cusparseIndexType_t */ compute_type,
            /* cusparseIndexBase_t */ CUSPARSE_INDEX_BASE_ZERO,
            /* cudaDataType */ data_type));

        // Create dense vector X
        PL_CUSPARSE_IS_SUCCESS(cusparseCreateDnVec(
            /* cusparseDnVecDescr_t* */ &vecX,
            /* int64_t */ num_cols,
            /* void* */ BaseType::getData(),
            /* cudaDataType */ data_type));

        // Create dense vector y
        PL_CUSPARSE_IS_SUCCESS(cusparseCreateDnVec(
            /* cusparseDnVecDescr_t* */ &vecY,
            /* int64_t */ num_rows,
            /* void* */ d_tmp.getData(),
            /* cudaDataType */ data_type));

        // allocate an external buffer if needed
        PL_CUSPARSE_IS_SUCCESS(cusparseSpMV_bufferSize(
            /* cusparseHandle_t */ handle,
            /* cusparseOperation_t */ CUSPARSE_OPERATION_NON_TRANSPOSE,
            /* const void* */ &alpha,
            /* cusparseSpMatDescr_t */ mat,
            /* cusparseDnVecDescr_t */ vecX,
            /* const void* */ &beta,
            /* cusparseDnVecDescr_t */ vecY,
            /* cudaDataType */ data_type,
            /* cusparseSpMVAlg_t */ CUSPARSE_SPMV_ALG_DEFAULT,
            /* size_t* */ &bufferSize));

        DataBuffer<cudaDataType_t, int> dBuffer{bufferSize, device_id,
                                                stream_id, true};

        // execute SpMV
        PL_CUSPARSE_IS_SUCCESS(cusparseSpMV(
            /* cusparseHandle_t */ handle,
            /* cusparseOperation_t */ CUSPARSE_OPERATION_NON_TRANSPOSE,
            /* const void* */ &alpha,
            /* cusparseSpMatDescr_t */ mat,
            /* cusparseDnVecDescr_t */ vecX,
            /* const void* */ &beta,
            /* cusparseDnVecDescr_t */ vecY,
            /* cudaDataType */ data_type,
            /* cusparseSpMVAlg_t */ CUSPARSE_SPMV_ALG_DEFAULT,
            /* void* */
            reinterpret_cast<void *>(dBuffer.getData())));

        // destroy matrix/vector descriptors
        PL_CUSPARSE_IS_SUCCESS(cusparseDestroySpMat(mat));
        PL_CUSPARSE_IS_SUCCESS(cusparseDestroyDnVec(vecX));
        PL_CUSPARSE_IS_SUCCESS(cusparseDestroyDnVec(vecY));

        expect = innerProdC_CUDA(BaseType::getData(), d_tmp.getData(),
                                 BaseType::getLength(), device_id, stream_id,
                                 getCublasCaller())
                     .x;

        return expect;
    }

  public:
    /**
     * @brief Utility method for probability calculation using given wires.
     *
     * @param wires List of wires to return probabilities for in lexicographical
     * order.
     * @return std::vector<double>
     */
    auto probability(const std::vector<size_t> &wires) -> std::vector<double> {
        // Data return type fixed as double in custatevec function call
        std::vector<double> probabilities(Util::exp2(wires.size()));
        // this should be built upon by the wires not participating
        int maskLen =
            0; // static_cast<int>(BaseType::getNumQubits() - wires.size());
        int *maskBitString = nullptr; //
        int *maskOrdering = nullptr;

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::vector<int> wires_int(wires.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(
            wires.begin(), wires.end(), wires_int.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        PL_CUSTATEVEC_IS_SUCCESS(custatevecAbs2SumArray(
            /* custatevecHandle_t */ handle_.get(),
            /* const void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ BaseType::getNumQubits(),
            /* double* */ probabilities.data(),
            /* const int32_t* */ wires_int.data(),
            /* const uint32_t */ wires_int.size(),
            /* const int32_t* */ maskBitString,
            /* const int32_t* */ maskOrdering,
            /* const uint32_t */ maskLen));

        return probabilities;
    }

    /**
     * @brief Utility method for expectation value calculations.
     *
     * @param obsName String label for observable. If already exists, will used
     * cached device value. If not, `gate_matrix` is expected, and will
     * automatically cache for future reuse.
     * @param wires Target wires for expectation value.
     * @param params Parameters for a parametric gate.
     * @param gate_matrix Optional matrix for observable. Caches for future use
     * if does not exist.
     * @return auto Expectation value.
     */
    auto expval(const std::string &obsName, const std::vector<size_t> &wires,
                const std::vector<Precision> &params = {0.0},
                const std::vector<CFP_t> &gate_matrix = {}) {

        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;
        auto &&local_wires =
            (gate_matrix.empty())
                ? wires
                : std::vector<size_t>{
                      wires.rbegin(),
                      wires.rend()}; // ensure wire indexing correctly preserved
                                     // for tensor-observables

        if (!(gate_cache_.gateExists(obsName, par[0]) || gate_matrix.empty())) {
            gate_cache_.add_gate(obsName, par[0], gate_matrix);
        } else if (!gate_cache_.gateExists(obsName, par[0]) &&
                   gate_matrix.empty()) {
            std::string message =
                "Currently unsupported observable: " + obsName;
            throw LightningException(message.c_str());
        }
        auto expect_val = getExpectationValueDeviceMatrix(
            gate_cache_.get_gate_device_ptr(obsName, par[0]), local_wires);
        return expect_val;
    }
    /**
     * @brief See `expval(const std::string &obsName, const std::vector<size_t>
     &wires, const std::vector<Precision> &params = {0.0}, const
     std::vector<CFP_t> &gate_matrix = {})`
     */
    auto expval(const std::string &obsName, const std::vector<size_t> &wires,
                const std::vector<Precision> &params = {0.0},
                const std::vector<std::complex<Precision>> &gate_matrix = {}) {
        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;

        std::vector<CFP_t> matrix_cu(gate_matrix.size());
        if (!(gate_cache_.gateExists(obsName, par[0]) || gate_matrix.empty())) {
            for (std::size_t i = 0; i < gate_matrix.size(); i++) {
                matrix_cu[i] = cuUtil::complexToCu<std::complex<Precision>>(
                    gate_matrix[i]);
            }
            gate_cache_.add_gate(obsName, par[0], matrix_cu);
        } else if (!gate_cache_.gateExists(obsName, par[0]) &&
                   gate_matrix.empty()) {
            std::string message =
                "Currently unsupported observable: " + obsName;
            throw LightningException(message.c_str());
        }
        return expval(obsName, wires, params, matrix_cu);
    }
    /**
     * @brief See `expval(std::vector<CFP_t> &gate_matrix = {})`
     */
    auto expval(const std::vector<size_t> &wires,
                const std::vector<std::complex<Precision>> &gate_matrix) {

        std::vector<CFP_t> matrix_cu(gate_matrix.size());

        for (std::size_t i = 0; i < gate_matrix.size(); i++) {
            matrix_cu[i] =
                cuUtil::complexToCu<std::complex<Precision>>(gate_matrix[i]);
        }

        if (gate_matrix.empty()) {
            std::string message = "Currently unsupported observable";
            throw LightningException(message.c_str());
        }

        // Wire order reversed to match expected custatevec wire ordering for
        // tensor observables.
        auto &&local_wires =
            (gate_matrix.empty())
                ? wires
                : std::vector<size_t>{wires.rbegin(), wires.rend()};

        auto expect_val =
            getExpectationValueDeviceMatrix(matrix_cu.data(), local_wires);
        return expect_val;
    }
};

} // namespace Pennylane::CUDA