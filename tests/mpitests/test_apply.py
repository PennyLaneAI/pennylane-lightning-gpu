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

numQubits = 6


def createRandomInitState(numWires):
    num_elements = 1 << numWires
    init_state = np.random.rand(num_elements) + 1j * np.random.rand(num_elements)
    scale_sum = np.sqrt(np.sum(np.abs(init_state) ** 2))
    init_state = init_state / scale_sum
    return init_state


class TestApply:
    # Parameterized test case for point-to-point communication
    @pytest.mark.parametrize(
        "data, expected_data",
        [
            ([0, 1, 2, 3], [3, 0, 1, 2]),
            ([1, 2, 3, 4], [4, 1, 2, 3]),
            ([10, 20, 30, 40], [40, 10, 20, 30]),
        ],
    )
    def test_senrecv(self, data, expected_data):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        local_data = comm.scatter(data, root=0)
        local_expected_data = comm.scatter(expected_data, root=0)

        # send data to the next process in the ring
        dest = (rank + 1) % size
        comm.send(local_data, dest=dest)

        # receive data from the previous process in the ring
        source = (rank - 1) % size
        received_data = comm.recv(source=source)
        # verify that the received data is correct
        assert received_data == local_expected_data

    test_single_qubit_gates_nonparam = [
        (qml.PauliX, [0]),
        (qml.PauliX, [1]),
        (qml.PauliX, [numQubits - 2]),
        (qml.PauliX, [numQubits - 1]),
        (qml.PauliY, [0]),
        (qml.PauliY, [1]),
        (qml.PauliY, [numQubits - 2]),
        (qml.PauliY, [numQubits - 1]),
        (qml.PauliZ, [0]),
        (qml.PauliZ, [1]),
        (qml.PauliZ, [numQubits - 2]),
        (qml.PauliZ, [numQubits - 1]),
        (qml.Hadamard, [0]),
        (qml.Hadamard, [1]),
        (qml.Hadamard, [numQubits - 2]),
        (qml.Hadamard, [numQubits - 1]),
        (qml.S, [0]),
        (qml.S, [1]),
        (qml.S, [numQubits - 2]),
        (qml.S, [numQubits - 1]),
        (qml.T, [0]),
        (qml.T, [1]),
        (qml.T, [numQubits - 2]),
        (qml.T, [numQubits - 1]),
    ]

    @pytest.mark.parametrize("operation, Wires", test_single_qubit_gates_nonparam)
    def test_apply_operation_single_wire_nonparam(self, tol, operation, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        mpi_comm = MPIManager(MPI.COMM_WORLD)
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        state_vector = np.zeros(1 << num_wires).astype(np.complex128)
        expected_output_cpu = np.zeros(1 << num_wires).astype(np.complex128)
        local_state_vector = np.zeros(1 << num_local_wires).astype(np.complex128)
        local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(np.complex128)

        state_vector = createRandomInitState(num_wires)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qml.device("default.qubit", wires=num_wires, c_dtype=np.complex128)

        @qml.qnode(dev_cpu)
        def circuit():
            qml.QubitStateVector(state_vector, wires=range(num_wires))
            operation(wires=Wires)
            return qml.state()

        expected_output_cpu = circuit()

        mpi_comm.Barrier

        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi_comm=comm, c_dtype=np.complex128
        )

        dev_gpumpi.syncH2D(local_state_vector)
        dev_gpumpi.apply([operation(wires=Wires)])
        dev_gpumpi.syncD2H(local_state_vector)

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    test_two_qubit_gates_nonparam = [
        (qml.CNOT, [0, 1]),
        (qml.CNOT, [numQubits - 2, numQubits - 1]),
        (qml.CNOT, [0, numQubits - 1]),
        (qml.SWAP, [0, 1]),
        (qml.SWAP, [numQubits - 2, numQubits - 1]),
        (qml.SWAP, [0, numQubits - 1]),
        (qml.CY, [0, 1]),
        (qml.CY, [numQubits - 2, numQubits - 1]),
        (qml.CY, [0, numQubits - 1]),
        (qml.CZ, [0, 1]),
        (qml.CZ, [numQubits - 2, numQubits - 1]),
        (qml.CZ, [0, numQubits - 1]),
    ]

    @pytest.mark.parametrize("operation, Wires", test_two_qubit_gates_nonparam)
    def test_apply_operation_two_wire_nonparam(self, tol, operation, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        mpi_comm = MPIManager(MPI.COMM_WORLD)
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        state_vector = np.zeros(1 << num_wires).astype(np.complex128)
        expected_output_cpu = np.zeros(1 << num_wires).astype(np.complex128)
        local_state_vector = np.zeros(1 << num_local_wires).astype(np.complex128)
        local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(np.complex128)

        state_vector = createRandomInitState(num_wires)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qml.device("default.qubit", wires=num_wires, c_dtype=np.complex128)

        @qml.qnode(dev_cpu)
        def circuit():
            qml.QubitStateVector(state_vector, wires=range(num_wires))
            operation(wires=Wires)
            return qml.state()

        expected_output_cpu = circuit()

        mpi_comm.Barrier
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi_comm=comm, c_dtype=np.complex128
        )

        dev_gpumpi.syncH2D(local_state_vector)
        dev_gpumpi.apply([operation(wires=Wires)])
        dev_gpumpi.syncD2H(local_state_vector)

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    test_three_qubit_gates_nonparam = [
        (qml.CSWAP, [0, 1, 2]),
        (qml.CSWAP, [numQubits - 3, numQubits - 2, numQubits - 1]),
        (qml.CSWAP, [0, 1, numQubits - 1]),
        (qml.CSWAP, [0, numQubits - 2, numQubits - 1]),
        (qml.Toffoli, [0, 1, 2]),
        (qml.Toffoli, [numQubits - 3, numQubits - 2, numQubits - 1]),
        (qml.Toffoli, [0, 1, numQubits - 1]),
        (qml.Toffoli, [0, numQubits - 2, numQubits - 1]),
    ]

    @pytest.mark.parametrize("operation, Wires", test_three_qubit_gates_nonparam)
    def test_apply_operation_three_wire_nonparam(self, tol, operation, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        mpi_comm = MPIManager(MPI.COMM_WORLD)
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        state_vector = np.zeros(1 << num_wires).astype(np.complex128)
        expected_output_cpu = np.zeros(1 << num_wires).astype(np.complex128)
        local_state_vector = np.zeros(1 << num_local_wires).astype(np.complex128)
        local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(np.complex128)

        state_vector = createRandomInitState(num_wires)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qml.device("default.qubit", wires=num_wires, c_dtype=np.complex128)

        @qml.qnode(dev_cpu)
        def circuit():
            qml.QubitStateVector(state_vector, wires=range(num_wires))
            operation(wires=Wires)
            return qml.state()

        expected_output_cpu = circuit()
        mpi_comm.Barrier
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi_comm=comm, c_dtype=np.complex128
        )

        dev_gpumpi.syncH2D(local_state_vector)
        dev_gpumpi.apply([operation(wires=Wires)])
        dev_gpumpi.syncD2H(local_state_vector)

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


# 1Qubits gate parametric
# PhaseShift, RX, RY, RZ, (single parameter)
# Rot (three parameters)

# 2 Qubit gate parametric
# CRX, CRY, CRZ, ControlledPhaseShift, SingleExcitation, SingleExcitationMinus, SingleExcitationPlus (single parameter)
# IsingXX, IsingYY, IsingZZ (single parameter)
# CRot(three parameters)

# 4 Qubit gate parametric
# DoubleExcitation, DoubleExcitationMinus, DoubleExcitationPlus, (single parameter)
