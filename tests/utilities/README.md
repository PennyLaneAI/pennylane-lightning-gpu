This folder holds some utility methods for GPU functionality.

# `sv_transfers.cu`: 
This allows sample profiling of transfering the statevector data between the device and host. Compile and run as:

```bash
nvcc ./sv_transfers.cu -o sv_transfers
nvprof ./sv_transfers <num_qubits>
``` 

where `<num_qubits>` is replaced by the nummber of qubits to examine transfer timings.
