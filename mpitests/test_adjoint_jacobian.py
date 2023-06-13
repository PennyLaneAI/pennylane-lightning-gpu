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
Tests for the ``adjoint_jacobian`` method of LightningGPU.
"""
from mpi4py import MPI
import itertools as it
import math
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode, qnode
from scipy.stats import unitary_group

from pennylane_lightning_gpu._serialize import _serialize_ob

try:
    from pennylane_lightning_gpu.lightning_gpu import CPP_BINARY_AVAILABLE

    if not CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU is unsupported on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU is unsupported on this platform. Skipping.",
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
        spy = mocker.spy(dev_gpumpi, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

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
        spy = mocker.spy(dev_gpumpi, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

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
        spy = mocker.spy(dev_gpumpi, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

        qnode2 = QNode(circuit, dev_gpumpi, diff_method="parameter-shift")
        grad_fn = qml.grad(qnode2)
        grad_PS = grad_fn(*args)

        assert np.allclose(grad_A, grad_PS, atol=tol, rtol=0)
