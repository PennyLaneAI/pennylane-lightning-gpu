# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane_lightning_gpu.LightningGPU` device (MPI).
"""
# pylint: disable=protected-access,cell-var-from-loop
from mpi4py import MPI
import math
import numpy as np
import pennylane as qml
import pytest
from pennylane import DeviceError

try:
    from pennylane_lightning_gpu.lightning_gpu import CPP_BINARY_AVAILABLE
    from pennylane_lightning_gpu import LightningGPU
    import pennylane_lightning_gpu as plg

    if not CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU binary is not found on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU binary is not found on this platform. Skipping.",
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
        "input_data",
        [
            np.array([0, 1, 2, 3]),
            np.array([1, 2, 3, 4]),
            np.array([10, 20, 30, 40]),
        ],
    )
    def test_sendrecv(self, input_data):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    
        local_data = comm.scatter(input_data, root=0)
        local_expected_data = comm.scatter(np.roll(input_data,1), root=0)

        if rank != 0:
            # receive data from the previous process in the ring
            received_data = comm.recv(source=(rank - 1 + size) % size)
        
        # send data to the next process in the ring
        comm.send(local_data, dest=(rank + 1) % size)

        if rank == 0:
            # receive data from the previous process in the ring 
            received_data = comm.recv(source=(rank - 1 + size) % size)

        # verify that the received data is correct
        assert received_data == local_expected_data

    # Parameterized test case for single wire nonparam gates
    @pytest.mark.parametrize("operation",[qml.PauliX,qml.PauliY,qml.PauliZ,qml.Hadamard,qml.S,qml.T])
    @pytest.mark.parametrize("Wires", [0,1,numQubits - 2,numQubits - 1])
    def test_apply_operation_single_wire_nonparam(self, tol, operation, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
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
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        dev_gpumpi.syncH2D(local_state_vector)
        dev_gpumpi.apply([operation(wires=Wires)])
        dev_gpumpi.syncD2H(local_state_vector)

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation", [qml.CNOT,qml.SWAP,qml.CY,qml.CZ])
    @pytest.mark.parametrize("Wires", [[0,1],[numQubits - 2, numQubits - 1],[0, numQubits - 1]])
    def test_apply_operation_two_wire_nonparam(self, tol, operation, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
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
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        dev_gpumpi.syncH2D(local_state_vector)
        dev_gpumpi.apply([operation(wires=Wires)])
        dev_gpumpi.syncD2H(local_state_vector)

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


    @pytest.mark.parametrize("operation", [qml.CSWAP,qml.Toffoli])
    @pytest.mark.parametrize("Wires", [[0, 1, 2],[numQubits - 3, numQubits - 2, numQubits - 1],[0, 1, numQubits - 1],[0, numQubits - 2, numQubits - 1]])
    def test_apply_operation_three_wire_nonparam(self, tol, operation, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
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
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        dev_gpumpi.syncH2D(local_state_vector)
        dev_gpumpi.apply([operation(wires=Wires)])
        dev_gpumpi.syncD2H(local_state_vector)

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation", [qml.CSWAP,qml.Toffoli])
    @pytest.mark.parametrize("Wires", [[0, 1, 2],[numQubits - 3, numQubits - 2, numQubits - 1],[0, 1, numQubits - 1],[0, numQubits - 2, numQubits - 1]])
    def test_apply_operation_three_wire_qnode_nonparam(self, tol, operation, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
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
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        @qml.qnode(dev_gpumpi)
        def circuit_mpi():
            qml.QubitStateVector(state_vector, wires=range(num_wires))
            operation(wires=Wires)
            return qml.state()

        local_state_vector = circuit_mpi()

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    test_gates_qnode_param = [
        (qml.PhaseShift, [0.1], [0]),
        (qml.PhaseShift, [0.1], [numQubits - 1]),
        (qml.RX, [0.2], [0]),
        (qml.RX, [0.2], [numQubits - 1]),
        (qml.RY, [0.3], [0]),
        (qml.RY, [0.3], [numQubits - 1]),
        (qml.RZ, [0.4], [0]),
        (qml.RZ, [0.4], [numQubits - 1]),
        (qml.Rot, [0.1, 0.2, 0.3], [0]),
        (qml.Rot, [0.1, 0.2, 0.3], [numQubits - 1]),
        (qml.CRX, [0.1], [0, 1]),
        (qml.CRX, [0.1], [0, numQubits - 1]),
        (qml.CRX, [0.1], [numQubits - 2, numQubits - 1]),
        (qml.CRY, [0.2], [0, 1]),
        (qml.CRY, [0.2], [0, numQubits - 1]),
        (qml.CRY, [0.2], [numQubits - 2, numQubits - 1]),
        (qml.CRZ, [0.3], [0, 1]),
        (qml.CRZ, [0.3], [0, numQubits - 1]),
        (qml.CRZ, [0.3], [numQubits - 2, numQubits - 1]),
        (qml.ControlledPhaseShift, [0.4], [0, 1]),
        (qml.ControlledPhaseShift, [0.4], [0, numQubits - 1]),
        (qml.ControlledPhaseShift, [0.4], [numQubits - 2, numQubits - 1]),
        (qml.SingleExcitation, [0.5], [0, 1]),
        (qml.SingleExcitation, [0.5], [0, numQubits - 1]),
        (qml.SingleExcitation, [0.5], [numQubits - 2, numQubits - 1]),
        (qml.SingleExcitationMinus, [0.6], [0, 1]),
        (qml.SingleExcitationMinus, [0.6], [0, numQubits - 1]),
        (qml.SingleExcitationMinus, [0.6], [numQubits - 2, numQubits - 1]),
        (qml.SingleExcitationPlus, [0.7], [0, 1]),
        (qml.SingleExcitationPlus, [0.7], [0, numQubits - 1]),
        (qml.SingleExcitationPlus, [0.7], [numQubits - 2, numQubits - 1]),
        (qml.IsingXX, [0.8], [0, 1]),
        (qml.IsingXX, [0.8], [0, numQubits - 1]),
        (qml.IsingXX, [0.8], [numQubits - 2, numQubits - 1]),
        (qml.IsingYY, [0.9], [0, 1]),
        (qml.IsingYY, [0.9], [0, numQubits - 1]),
        (qml.IsingYY, [0.9], [numQubits - 2, numQubits - 1]),
        (qml.IsingZZ, [0.1], [0, 1]),
        (qml.IsingZZ, [0.1], [0, numQubits - 1]),
        (qml.IsingZZ, [0.1], [numQubits - 2, numQubits - 1]),
        (qml.CRot, [0.1, 0.2, 0.3], [0, 1]),
        (qml.CRot, [0.1, 0.2, 0.3], [0, numQubits - 1]),
        (qml.CRot, [0.1, 0.2, 0.3], [numQubits - 2, numQubits - 1]),
        (qml.DoubleExcitation, [0.1], [0, 1, 2, 3]),
        (qml.DoubleExcitation, [0.1], [0, 1, numQubits - 2, numQubits - 1]),
        (qml.DoubleExcitationPlus, [0.2], [0, 1, 2, 3]),
        (qml.DoubleExcitationPlus, [0.2], [0, 1, numQubits - 2, numQubits - 1]),
        (qml.DoubleExcitationMinus, [0.3], [0, 1, 2, 3]),
        (qml.DoubleExcitationMinus, [0.3], [0, 1, numQubits - 2, numQubits - 1]),
    ]

    @pytest.mark.parametrize("operation, par, Wires", test_gates_qnode_param)
    def test_apply_operation_gates_qnode_nonparam(self, tol, operation, par, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
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
            operation(*par, wires=Wires)
            return qml.state()

        expected_output_cpu = circuit()
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        @qml.qnode(dev_gpumpi)
        def circuit_mpi():
            qml.QubitStateVector(state_vector, wires=range(num_wires))
            operation(*par, wires=Wires)
            return qml.state()

        local_state_vector = circuit_mpi()

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    #BasisState test
    @pytest.mark.parametrize("operation", [qml.BasisState])
    @pytest.mark.parametrize("index", range(numQubits))
    def test_state_prep(self, tol, operation, index):
        par = np.zeros(numQubits,dtype=int)
        par[index]=1
        num_wires = numQubits
        comm = MPI.COMM_WORLD
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
            operation(par, wires=range(numQubits))
            return qml.state()

        expected_output_cpu = circuit()
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        @qml.qnode(dev_gpumpi)
        def circuit_mpi():
            operation(par, wires=range(numQubits))
            return qml.state()

        local_state_vector = circuit_mpi()

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    test_qubit_state_prep = [
        (qml.QubitStateVector, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [0]),
        (qml.QubitStateVector, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [1]),
        (qml.QubitStateVector, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [2]),
        (qml.QubitStateVector, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [3]),
        (qml.QubitStateVector, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [4]),
        (qml.QubitStateVector, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [5]),
        (qml.QubitStateVector, np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [1, 0]),
        (qml.QubitStateVector, np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [0, 1]),
        (qml.QubitStateVector, np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [0, 2]),
        (
            qml.QubitStateVector,
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [numQubits - 2, numQubits - 1],
        ),
        (
            qml.QubitStateVector,
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [0, numQubits - 1],
        ),
        (
            qml.QubitStateVector,
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [0, numQubits - 2],
        ),
    ]

    @pytest.mark.parametrize("operation, par, Wires", test_qubit_state_prep)
    def test_qubit_state_prep(self, tol, operation, par, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
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
            operation(par, wires=Wires)
            return qml.state()

        expected_output_cpu = circuit()
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        @qml.qnode(dev_gpumpi)
        def circuit_mpi():
            operation(par, wires=Wires)
            return qml.state()

        local_state_vector = circuit_mpi()
        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    test_dev_reset = [
        (qml.QubitStateVector, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [0]),
    ]

    @pytest.mark.parametrize("operation, par, Wires", test_dev_reset)
    def test_dev_reset(self, tol, operation, par, Wires):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
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

        dev_cpu.reset()

        @qml.qnode(dev_cpu)
        def circuit():
            return qml.state()

        expected_output_cpu = circuit()
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        dev_cpu.reset()

        @qml.qnode(dev_gpumpi)
        def circuit_mpi():
            return qml.state()

        local_state_vector = circuit_mpi()
        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)
