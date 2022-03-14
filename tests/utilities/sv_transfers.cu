#include <complex>
#include <cstddef>
#include <cuComplex.h>
#include <cuda.h>
#include <iostream>
#include <vector>

/**
 * @brief Simple utility to transfer data from device to host or vice-versa.
 * Adapted from
 * https://www.microway.com/hpc-tech-tips/cuda-host-to-device-transfers-and-data-movement/
 *
 * @param num_qubits
 */

template <typename DataType = cuDoubleComplex>
void transferTimings(std::size_t num_qubits) {
    const std::size_t num_elements = (0b1 << num_qubits);
    const std::size_t data_size = num_elements * sizeof(DataType);

    std::vector<DataType> data_host(num_elements, {0, 0});
    DataType *data_device;
    cudaMalloc((void **)&data_device, data_size);

    cudaMemcpy(data_device, data_host.data(), data_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(data_host.data(), data_device, data_size,
               cudaMemcpyDeviceToHost);

    cudaFree(data_device);
}

int main(int argc, char *argv[]) {
    std::size_t num_qubits = std::atoi(argv[1]);
    auto b = (0b1 << num_qubits);
    std::cout << "Using " << num_qubits << " qubits totaling " << b << " bytes."
              << std::endl;
    transferTimings<cuDoubleComplex>(num_qubits);
}
