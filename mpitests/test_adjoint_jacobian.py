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
Tests for the ``adjoint_jacobian`` method of LightningGPU with MPI support.
"""
from mpi4py import MPI
import numpy as np
import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode, qnode
import pytest
from scipy.stats import unitary_group



try:
    from pennylane_lightning_gpu.lightning_gpu import CPP_BINARY_AVAILABLE

    if not CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU binary is not found on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU binary is not found on this platform. Skipping.",
        allow_module_level=True,
    )

I, X, Y, Z = (
    np.eye(2),
    qml.PauliX.compute_matrix(),
    qml.PauliY.compute_matrix(),
    qml.PauliZ.compute_matrix(),
)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    def test_qnode_name_obs(self, mocker, tol):
        """Test that specifying diff_method allows the adjoint method to be selected"""

        num_wires = 3
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        commSize = comm.Get_size()

        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        dev_gpumpi = qml.device("lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128)

        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(x, y, z, wires=0)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        qnode1 = QNode(circuit, dev_gpumpi, diff_method="adjoint")
        #spy = mocker.spy(dev_gpumpi, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        #spy.assert_called()

        qnode2 = QNode(circuit, dev_gpumpi, diff_method="parameter-shift")
        grad_fn = qml.grad(qnode2)
        grad_PS = grad_fn(*args)

        assert np.allclose(grad_A, grad_PS, atol=tol, rtol=0)

    def test_qnode_tensor_obs(self, mocker, tol):
        """Test that specifying diff_method allows the adjoint method to be selected"""

        num_wires = 3
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        commSize = comm.Get_size()

        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        dev_gpumpi = qml.device("lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128)

        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(x, y, z, wires=0)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        qnode1 = QNode(circuit, dev_gpumpi, diff_method="adjoint")
        #spy = mocker.spy(dev_gpumpi, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        #spy.assert_called()

        qnode2 = QNode(circuit, dev_gpumpi, diff_method="parameter-shift")
        grad_fn = qml.grad(qnode2)
        grad_PS = grad_fn(*args)

        assert np.allclose(grad_A, grad_PS, atol=tol, rtol=0)

    def test_qnode_ham_obs(self, mocker, tol):
        """Test that specifying diff_method allows the adjoint method to be selected"""

        num_wires = 3
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        commSize = comm.Get_size()

        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        dev_gpumpi = qml.device("lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128)

        obs = qml.PauliX(1) @ qml.PauliY(2)
        obs1 = qml.Identity(1)
        H = qml.Hamiltonian([1.0, 1.0], [obs1, obs])

        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(x, y, z, wires=0)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(H)

        qnode1 = QNode(circuit, dev_gpumpi, diff_method="adjoint")
        #spy = mocker.spy(dev_gpumpi, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        #spy.assert_called()

        qnode2 = QNode(circuit, dev_gpumpi, diff_method="parameter-shift")
        grad_fn = qml.grad(qnode2)
        grad_PS = grad_fn(*args)

        assert np.allclose(grad_A, grad_PS, atol=tol, rtol=0)

def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qml.QubitStateVector(unitary_group.rvs(2**6, random_state=0)[0], wires=wires)
    qml.RX(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.adjoint(qml.RX(params[2], wires=wires[2]))
    qml.RZ(params[0], wires=wires[3])
    qml.CRX(params[3], wires=[wires[3], wires[0]])
    qml.PhaseShift(params[4], wires=wires[2])
    qml.CRY(params[5], wires=[wires[2], wires[1]])
    qml.adjoint(qml.CRZ(params[5], wires=[wires[0], wires[3]]))
    qml.adjoint(qml.PhaseShift(params[6], wires=wires[0]))
    qml.Rot(params[6], params[7], params[8], wires=wires[0])
    qml.adjoint(qml.Rot(params[8], params[8], params[9], wires=wires[1]))
    qml.MultiRZ(params[11], wires=[wires[0], wires[1]])
    qml.CPhase(params[12], wires=[wires[3], wires[2]])
    qml.IsingXX(params[13], wires=[wires[1], wires[0]])
    qml.IsingYY(params[14], wires=[wires[3], wires[2]])
    qml.IsingZZ(params[15], wires=[wires[2], wires[1]])
    qml.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qml.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])


@pytest.mark.parametrize(
    "returns",
    [
        (qml.PauliX(0),),
        (qml.PauliY(0),),
        (qml.PauliZ(0),),
        (qml.PauliX(1),),
        (qml.PauliY(1),),
        (qml.PauliZ(1),),
        (qml.PauliX(2),),
        (qml.PauliY(2),),
        (qml.PauliZ(2),),
        (qml.PauliX(3),),
        (qml.PauliY(3),),
        (qml.PauliZ(3),),
        (qml.PauliX(0), qml.PauliY(1)),
        (
            qml.PauliZ(0),
            qml.PauliX(1),
            qml.PauliY(2),
        ),
        (
            qml.PauliY(0),
            qml.PauliZ(1),
            qml.PauliY(3),
        ),
        (qml.PauliZ(0) @ qml.PauliY(3),),
        (qml.Hadamard(2),),
        (qml.Hadamard(3) @ qml.PauliZ(2),),
        (qml.PauliX(0) @ qml.PauliY(3),),
        (qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliY(3),),
        (qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2),),
        #(0.5 * qml.PauliZ(0) @ qml.PauliZ(2),),
    ],
)
def test_integration(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    num_wires = 6
    comm = MPI.COMM_WORLD
    dev_default = qml.device("default.qubit", wires=range(num_wires))
    dev_gpu = qml.device("lightning.gpu", wires=range(num_wires), mpi=True, c_dtype=np.complex128)

    def circuit(params):
        circuit_ansatz(params, wires=range(num_wires))
        return [qml.expval(r) for r in returns]

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_default = qml.QNode(circuit, dev_default, diff_method="parameter-shift")

    def convert_to_array_gpu(params):
        return np.array(qnode_gpu(params))

    def convert_to_array_default(params):
        return np.array(qnode_default(params))

    j_gpu = qml.jacobian(convert_to_array_gpu)(params)
    j_default = qml.jacobian(convert_to_array_default)(params)

    assert np.allclose(j_gpu, j_default, atol=1e-7)


custom_wires = ["alice", 3.14, -1, 0, "bob", "luc"]


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(custom_wires[0]),
        qml.PauliX(custom_wires[2]),
        qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.Hadamard(custom_wires[2]),
        qml.Hadamard(custom_wires[3]) @ qml.PauliZ(custom_wires[2]),
        qml.PauliX(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.PauliY(custom_wires[0]) @ qml.PauliY(custom_wires[2]) @ qml.PauliY(custom_wires[3]),
    ],
)
def test_integration_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""
    comm = MPI.COMM_WORLD
    dev_lightning = qml.device("lightning.qubit", wires=custom_wires)
    dev_gpu = qml.device("lightning.gpu", wires=custom_wires, mpi=True, c_dtype=np.complex128)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns), qml.expval(qml.PauliY(custom_wires[1]))

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="parameter-shift")

    def convert_to_array_gpu(params):
        return np.array(qnode_gpu(params))

    def convert_to_array_lightning(params):
        return np.array(qnode_lightning(params))

    j_gpu = qml.jacobian(convert_to_array_gpu)(params)
    j_lightning = qml.jacobian(convert_to_array_lightning)(params)

    assert np.allclose(j_gpu, j_lightning, atol=1e-7)