// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved

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
 * @file init.cu
 */

#include "cuda_helpers.hpp"
#include <cuComplex.h>

namespace {

template <typename T> __global__ void init_sv_kernel(T *x, int64_t n) {
    int64_t const i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i == 0) {
        x[i].x = 1.0;
        x[i].y = 0.0;
    } else if (i < n) {
        x[i].x = 0.0;
        x[i].y = 0.0;
    }
}

template <typename T>
void call_init_kernel(T *x, int64_t n, cudaStream_t stream) {
    constexpr int64_t threads = 256;
    const int64_t blocks = n / threads + (n % threads == 0 ? 0 : 1);
    init_sv_kernel<<<blocks, threads, 0, stream>>>(x, n);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}
} // namespace

namespace Pennylane {
void initialize_sv(cuComplex *x, int64_t n, cudaStream_t stream) {
    call_init_kernel(x, n, stream);
}

void initialize_sv(cuDoubleComplex *x, int64_t n, cudaStream_t stream) {
    call_init_kernel(x, n, stream);
}
} // namespace Pennylane
