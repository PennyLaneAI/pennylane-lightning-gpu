# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the :mod:`pennylane_lightning_gpu.LightningGPU` device.
"""
# pylint: disable=protected-access,cell-var-from-loop
import math
from mpi4py import MPI
import numpy as np
import pennylane as qml
import pytest
from pennylane import DeviceError

try:
    from pennylane_lightning_gpu.lightning_gpu import CPP_BINARY_AVAILABLE
    from pennylane_lightning_gpu.lightning_gpu_qubit_ops import MPIManager, DevPool
    from pennylane_lightning_gpu import LightningGPU
    import pennylane_lightning_gpu as plg

    if not CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU is unsupported on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU is unsupported on this platform. Skipping.",
        allow_module_level=True,
    )


class TestApply:
    # Parameterized test case for point-to-point communication
    @pytest.mark.parametrize("data, expected_data", [
        ([0, 1, 2, 3], [3, 0, 1, 2]),
        ([1, 2, 3, 4], [4, 1, 2, 3]),
        ([10, 20, 30, 40], [40, 10, 20, 30])
    ])
    def test_senrecv(self, data, expected_data):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        local_data = comm.scatter(data, root=0)
        local_expected_data = comm.scatter(expected_data,root=0)

        # send data to the next process in the ring
        dest = (rank + 1) % size
        comm.send(local_data, dest=dest)

        # receive data from the previous process in the ring
        source = (rank - 1) % size
        received_data = comm.recv(source=source)
        # verify that the received data is correct
        assert received_data == local_expected_data

    
    test_data_no_parameters = [
        (qml.PauliX, 
        np.array(
        [0.1653855288944372 + 0.08360762242222763j,
         0.0731293375604395 + 0.13209080879903976j,
         0.23742759434160687 + 0.2613440813782711j,
         0.16768740742688235 + 0.2340607179431313j,
         0.2247465091396771 + 0.052469062762363974j,
         0.1595307101966878 + 0.018355977199570113j,
         0.01433428625707798 + 0.18836803047905595j,
         0.20447553584586473 + 0.02069817884076428j,
         0.17324175995006008 + 0.12834320562185453j,
         0.021542232643170886 + 0.2537776554975786j,
         0.2917899745322105 + 0.30227665008366594j,
         0.17082687702494623 + 0.013880922806771745j,
         0.03801974084659355 + 0.2233816291263903j,
         0.1991010562067874 + 0.2378546697582974j,
         0.13833362414043807 + 0.0571737109901294j,
         0.1960850292216881 + 0.22946370987301284j],dtype=np.complex128), 
        np.array(
            [0.17324175995006008 + 0.12834320562185453j,
            0.021542232643170886 + 0.2537776554975786j,
            0.2917899745322105 + 0.30227665008366594j,
            0.17082687702494623 + 0.013880922806771745j,
            0.03801974084659355 + 0.2233816291263903j,
            0.1991010562067874 + 0.2378546697582974j,
            0.13833362414043807 + 0.0571737109901294j,
            0.1960850292216881 + 0.22946370987301284j,
            0.1653855288944372 + 0.08360762242222763j,
            0.0731293375604395 + 0.13209080879903976j,
            0.23742759434160687 + 0.2613440813782711j,
            0.16768740742688235 + 0.2340607179431313j, 
            0.2247465091396771 + 0.052469062762363974j,
            0.1595307101966878 + 0.018355977199570113j,
            0.01433428625707798 + 0.18836803047905595j,
            0.20447553584586473 + 0.02069817884076428j,],dtype=np.complex128)),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
    def test_apply_operation_single_wire_no_parameters(
        self, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        comm = MPI.COMM_WORLD
        dev = LightningGPU(wires=4, mpi_comm=comm, c_dtype=np.complex128)
        #dev = qml.device('lightning.gpu', wires=4, mpi_comm=comm, c_dtype=np.complex128)
        
        state_vector = np.zeros(2**2).astype(dev.C_DTYPE)
        comm.Scatter(input, state_vector, root=0)
        
        DevPool.syncDevice
        dev.syncH2D(state_vector)
        DevPool.syncDevice
        comm.Barrier()

        dev.apply([operation(wires=[0])])
        DevPool.syncDevice
        comm.Barrier()


        dev.syncD2H(state_vector)
        DevPool.syncDevice

        comm.Barrier()

        local_expected_output = np.empty(2**2, dtype=np.complex128)
        comm.Scatter(expected_output, local_expected_output, root=0)
        
        comm.Barrier()


        assert np.allclose(state_vector, local_expected_output, atol=tol, rtol=0)

    
