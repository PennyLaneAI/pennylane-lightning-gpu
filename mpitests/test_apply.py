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

numQubits = 8

def createRandomInitState(numWires, seed_value=48):
    np.random.seed(seed_value)
    num_elements = 1 << numWires
    init_state = np.random.rand(num_elements) + 1j * np.random.rand(num_elements)
    scale_sum = np.sqrt(np.sum(np.abs(init_state) ** 2))
    init_state = init_state / scale_sum
    return init_state

def apply_operation_gates_qnode_param(tol, operation, par, Wires):
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
    
def apply_operation_gates_apply_param(tol, operation, par, Wires):
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

    dev_gpumpi.syncH2D(local_state_vector)
    dev_gpumpi.apply([operation(*par,wires=Wires)])
    dev_gpumpi.syncD2H(local_state_vector)

    assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

def apply_operation_gates_qnode_nonparam(tol, operation, Wires):
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
    
def apply_operation_gates_apply_nonparam(tol, operation, Wires):
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
    
class TestApply:
    # Parameterized test case for single wire nonparam gates
    @pytest.mark.parametrize("operation",[qml.PauliX,qml.PauliY,qml.PauliZ,qml.Hadamard,qml.S,qml.T])
    @pytest.mark.parametrize("Wires", [0,1,numQubits - 2,numQubits - 1])
    def test_apply_operation_single_wire_nonparam(self, tol, operation, Wires):
        apply_operation_gates_qnode_nonparam(tol, operation, Wires)
        apply_operation_gates_apply_nonparam(tol, operation, Wires)

    @pytest.mark.parametrize("operation", [qml.CNOT,qml.SWAP,qml.CY,qml.CZ])
    @pytest.mark.parametrize("Wires", [[0,1],[numQubits - 2, numQubits - 1],[0, numQubits - 1]])
    def test_apply_operation_two_wire_nonparam(self, tol, operation, Wires):
        apply_operation_gates_qnode_nonparam(tol, operation, Wires)
        apply_operation_gates_apply_nonparam(tol, operation, Wires)

    @pytest.mark.parametrize("operation", [qml.CSWAP,qml.Toffoli])
    @pytest.mark.parametrize("Wires", [[0, 1, 2],[numQubits - 3, numQubits - 2, numQubits - 1],[0, 1, numQubits - 1],[0, numQubits - 2, numQubits - 1]])
    def test_apply_operation_three_wire_nonparam(self, tol, operation, Wires):
        apply_operation_gates_qnode_nonparam(tol, operation, Wires)
        apply_operation_gates_apply_nonparam(tol, operation, Wires)

    @pytest.mark.parametrize("operation", [qml.CSWAP,qml.Toffoli])
    @pytest.mark.parametrize("Wires", [[0, 1, 2],[numQubits - 3, numQubits - 2, numQubits - 1],[0, 1, numQubits - 1],[0, numQubits - 2, numQubits - 1]])
    def test_apply_operation_three_wire_qnode_nonparam(self, tol, operation, Wires):
        apply_operation_gates_qnode_nonparam(tol, operation, Wires)
        apply_operation_gates_apply_nonparam(tol, operation, Wires)

    @pytest.mark.parametrize("operation", [qml.PhaseShift,qml.RX,qml.RY,qml.RZ])
    @pytest.mark.parametrize("par", [[0.1],[0.2],[0.3]])
    @pytest.mark.parametrize("Wires", [0,numQubits - 1])
    def test_apply_operation_1gatequbit_1param_gate_qnode_param(self, tol, operation, par, Wires):
        apply_operation_gates_qnode_param(tol, operation, par, Wires)
        apply_operation_gates_apply_param(tol, operation, par, Wires)

    @pytest.mark.parametrize("operation", [qml.Rot])
    @pytest.mark.parametrize("par", [[0.1,0.2,0.3],[0.2,0.3,0.4]])
    @pytest.mark.parametrize("Wires", [0,numQubits - 1])
    def test_apply_operation_1gatequbit_3param_gate_qnode_param(self, tol, operation, par, Wires):
        apply_operation_gates_qnode_param(tol, operation, par, Wires)
        apply_operation_gates_apply_param(tol, operation, par, Wires)

    @pytest.mark.parametrize("operation", [qml.CRot])
    @pytest.mark.parametrize("par", [[0.1,0.2,0.3],[0.2,0.3,0.4]])
    @pytest.mark.parametrize("Wires", [[0,numQubits - 1],[0,1],[numQubits - 2,numQubits - 1]])
    def test_apply_operation_1gatequbit_3param_gate_qnode_param(self, tol, operation, par, Wires):
        apply_operation_gates_qnode_param(tol, operation, par, Wires)
        apply_operation_gates_apply_param(tol, operation, par, Wires)

    @pytest.mark.parametrize("operation", [qml.CRX, qml.CRY, qml.CRZ, qml.ControlledPhaseShift,qml.SingleExcitation, qml.SingleExcitationMinus, qml.SingleExcitationPlus,qml.IsingXX,qml.IsingYY,qml.IsingZZ])
    @pytest.mark.parametrize("par", [[0.1],[0.2],[0.3]])
    @pytest.mark.parametrize("Wires", [[0,numQubits - 1],[0,1],[numQubits - 2,numQubits - 1]])
    def test_apply_operation_2gatequbit_1param_gate_qnode_param(self, tol, operation, par, Wires):
        apply_operation_gates_qnode_param(tol, operation, par, Wires)
        apply_operation_gates_apply_param(tol, operation, par, Wires)
    
    @pytest.mark.parametrize("operation", [qml.DoubleExcitation,qml.DoubleExcitationMinus,qml.DoubleExcitationPlus])
    @pytest.mark.parametrize("par", [[0.13],[0.2],[0.3]])
    @pytest.mark.parametrize("Wires", [[0,1,numQubits - 2,numQubits - 1],[0,1,2,3],[numQubits - 4,numQubits - 3,numQubits - 2,numQubits - 1]])
    def test_apply_operation_4gatequbit_1param_gate_qnode_param(self, tol, operation, par, Wires):
        apply_operation_gates_qnode_param(tol, operation, par, Wires)
        apply_operation_gates_apply_param(tol, operation, par, Wires)

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
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [0]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [1]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [2]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [3]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [4]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [5]),
        (np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [1, 0]),
        (np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [0, 1]),
        (np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [0, 2]),
        (
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [numQubits - 2, numQubits - 1],
        ),
        (
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [0, numQubits - 1],
        ),
        (
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [0, numQubits - 2],
        ),
    ]

    @pytest.mark.parametrize("par, Wires", test_qubit_state_prep)
    def test_qubit_state_prep(self, tol, par, Wires):
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
            qml.QubitStateVector(par, wires=Wires)
            return qml.state()

        expected_output_cpu = circuit()
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        @qml.qnode(dev_gpumpi)
        def circuit_mpi():
            qml.QubitStateVector(par, wires=Wires)
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


        @qml.qnode(dev_cpu)
        def circuit():
            return qml.state()

        expected_output_cpu = circuit()
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128
        )

        @qml.qnode(dev_gpumpi)
        def circuit_mpi():
            return qml.state()

        local_state_vector = circuit_mpi()
        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)
