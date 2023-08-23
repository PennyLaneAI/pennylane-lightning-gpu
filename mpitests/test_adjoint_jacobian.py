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
import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode, qnode
import pytest
from scipy.stats import unitary_group

import itertools as it
import math

from pennylane_lightning_gpu.lightning_gpu_qubit_ops import (
    NamedObsGPUMPI_C64,
    NamedObsGPUMPI_C128,
    TensorProdObsGPUMPI_C64,
    TensorProdObsGPUMPI_C128,
    HamiltonianGPUMPI_C64,
    HamiltonianGPUMPI_C128,
    SparseHamiltonianGPUMPI_C64,
    SparseHamiltonianGPUMPI_C128,
)
from pennylane_lightning_gpu._serialize import _serialize_ob


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


class TestAdjointJacobian:
    """Test QNode integration with the adjoint_jacobian method"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_not_expval_adj_mpi(self, isBatch_obs, request):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""
        num_wires = 8
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            dev_gpumpi.adjoint_jacobian(tape)

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_finite_shots_warns_adj_mpi(self, isBatch_obs):
        """Tests warning raised when finite shots specified"""

        dev_gpumpi = qml.device("lightning.gpu", wires=1, shots=1, batch_obs=isBatch_obs)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev_gpumpi.adjoint_jacobian(tape)

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_unsupported_op(self, isBatch_obs):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""
        num_wires = 8
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The CRot operation is not supported using the",
        ):
            dev_gpumpi.adjoint_jacobian(tape)

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_proj_unsupported(self, isBatch_obs):
        """Test if a QuantumFunctionError is raised for a Projector observable"""
        num_wires = 8
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev_gpumpi.adjoint_jacobian(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev_gpumpi.adjoint_jacobian(tape)

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_unsupported_hermitian_expectation(self, isBatch_obs):
        """Test if a QuantumFunctionError is raised for a Hermitian observable"""
        num_wires = 8
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        obs = np.array([[1, 0], [0, -1]], dtype=np.complex128, requires_grad=False)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.1, wires=(0,))
            qml.expval(qml.Hermitian(obs, wires=(0,)))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="LightningGPU adjoint differentiation method does not",
        ):
            dev_gpumpi.adjoint_jacobian(tape)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.1, wires=(0,))
            qml.expval(qml.Hermitian(obs, wires=(0,)) @ qml.PauliZ(wires=1))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="LightningGPU adjoint differentiation method does not",
        ):
            dev_gpumpi.adjoint_jacobian(tape)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_pauli_rotation_gradient(self, stateprep, G, theta, tol, isBatch_obs, request):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )
        dev_cpu = qml.device("default.qubit", wires=3)

        with qml.tape.QuantumTape() as tape:
            stateprep(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        calculated_val = dev_gpumpi.adjoint_jacobian(tape)
        expected_val = dev_cpu.adjoint_jacobian(tape)

        assert np.allclose(calculated_val, expected_val, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_Rot_gradient(self, stateprep, theta, tol, isBatch_obs, request):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )
        dev_cpu = qml.device("default.qubit", wires=3)

        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            stateprep(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        calculated_val = dev_gpumpi.adjoint_jacobian(tape)
        expected_val = dev_cpu.adjoint_jacobian(tape)

        assert np.allclose(calculated_val, expected_val, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_ry_gradient(self, par, tol, isBatch_obs, request):
        """Test that the gradient of the RY gate matches the exact analytic formula."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS = fn(qml.execute(gtapes, dev_gpumpi, gradient_fn=None))
        grad_A = dev_gpumpi.adjoint_jacobian(tape)

        # different methods must agree
        assert np.allclose(grad_PS, exact, atol=tol, rtol=0)
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_rx_gradient(self, tol, isBatch_obs, request):
        """Test that the gradient of the RX gate matches the known formula."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        a = 0.7418

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        # circuit jacobians
        dev_jacobian = dev_gpumpi.adjoint_jacobian(tape)
        expected_jacobian = -np.sin(a)
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_multiple_rx_gradient(self, tol, isBatch_obs, request):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        # circuit jacobians
        grad_A_gpu = dev_gpumpi.adjoint_jacobian(tape)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS_gpu = fn(qml.execute(gtapes, dev_gpumpi, gradient_fn=None))

        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(grad_PS_gpu, grad_A_gpu, atol=tol, rtol=0)
        assert np.allclose(grad_A_gpu, expected_jacobian, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("obs", [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity])
    @pytest.mark.parametrize(
        "op",
        [
            qml.RX(0.4, wires=0),
            qml.RY(0.6, wires=0),
            qml.RZ(0.8, wires=0),
            qml.CRX(1.0, wires=[0, 1]),
            qml.CRY(2.0, wires=[0, 1]),
            qml.CRZ(3.0, wires=[0, 1]),
            qml.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_gradients(self, op, obs, tol, isBatch_obs, request):
        """Tests that the gradients of circuits match between the param-shift and device
        methods."""

        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        # op.num_wires and op.num_params must be initialized a priori
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.apply(op)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.adjoint(qml.RY(0.5, wires=1), lazy=False)
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        tape.trainable_params = set(range(1, 1 + op.num_params))

        grad_PS = (lambda t, fn: fn(qml.execute(t, dev_gpumpi, None)))(
            *qml.gradients.param_shift(tape)
        )
        grad_D = dev_gpumpi.adjoint_jacobian(tape)

        assert np.allclose(grad_D, grad_PS, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_gradient_gate_with_multiple_parameters(self, tol, isBatch_obs, request):
        """Tests that gates with multiple free parameters yield correct gradients."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        grad_D = dev_gpumpi.adjoint_jacobian(tape)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS = fn(qml.execute(gtapes, dev_gpumpi, gradient_fn=None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_PS, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_use_device_state(self, tol, isBatch_obs, request):
        """Tests that when using the device state, the correct answer is still returned."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev_gpumpi.adjoint_jacobian(tape)

        qml.execute([tape], dev_gpumpi, None)
        dM2 = dev_gpumpi.adjoint_jacobian(tape, use_device_state=True)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_provide_starting_state(self, tol, isBatch_obs, request):
        """Tests provides correct answer when provided starting state."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev_gpumpi.adjoint_jacobian(tape)

        qml.execute([tape], dev_gpumpi, None)
        local_state_vector = np.zeros(len(dev_gpumpi.state)).astype(dev_gpumpi.C_DTYPE)
        dev_gpumpi.syncD2H(local_state_vector)
        dM2 = dev_gpumpi.adjoint_jacobian(tape, starting_state=local_state_vector)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize(
        "old_obs",
        [
            qml.PauliX(0) @ qml.PauliZ(1),
            qml.Hamiltonian([1.1], [qml.PauliZ(0)]),
            qml.Hamiltonian([1.1, 2.2], [qml.PauliZ(0), qml.PauliZ(1)]),
            qml.Hamiltonian([1.1, 2.2], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliX(1)]),
        ],
    )
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_op_arithmetic_is_supported(self, old_obs, isBatch_obs, tol, request):
        """Tests that an arithmetic obs with a PauliRep are supported for adjoint_jacobian."""

        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        def run_circuit(obs):
            params = qml.numpy.array([1.1, 2.2, 0.66, 1.23])

            @qml.qnode(dev_gpumpi, diff_method="adjoint")
            def circuit(par):
                qml.RX(par[0], 0)
                qml.RY(par[1], 0)
                qml.RX(par[2], 1)
                qml.RY(par[3], 1)
                return qml.expval(obs)

            return qml.jacobian(circuit)(params)

        new_obs = qml.pauli.pauli_sentence(old_obs).operation()
        res_old = run_circuit(old_obs)
        res_new = run_circuit(new_obs)
        assert np.allclose(res_old, res_new, atol=tol, rtol=0)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_finite_shots_warning(self, isBatch_obs):
        """Tests that a warning is raised when computing the adjoint diff on a device with finite shots"""

        dev = qml.device("lightning.gpu", wires=2, mpi=True, shots=1, batch_obs=isBatch_obs)
        param = qml.numpy.array(0.1)

        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):

            @qml.qnode(dev, diff_method="adjoint")
            def circ(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            qml.grad(circ)(param)

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_qnode(self, mocker, tol, isBatch_obs, request):
        """Test that specifying diff_method allows the adjoint method to be selected"""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

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

    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_gradient_repeated_gate_parameters(self, mocker, tol, isBatch_obs, request):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qml.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qml.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qml.expval(qml.PauliX(0))

        spy_analytic = mocker.spy(dev_gpumpi, "adjoint_jacobian")

        cost = QNode(circuit, dev_gpumpi, diff_method="parameter-shift")

        grad_fn = qml.grad(cost)
        grad_PS = grad_fn(params)

        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev_gpumpi, diff_method="adjoint")
        grad_fn = qml.grad(cost)
        grad_D = grad_fn(params)

        spy_analytic.assert_called_once()

        # the different methods agree
        assert np.allclose(grad_D, grad_PS, atol=tol, rtol=0)

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_interface_tf(self, isBatch_obs):
        """Test if gradients agree between the adjoint and parameter-shift methods when using the
        TensorFlow interface"""
        tf = pytest.importorskip("tensorflow")

        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * tf.sqrt(params2), wires=[0])
            qml.RY(tf.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = tf.Variable(0.3, dtype=tf.float64)
        params2 = tf.Variable(0.4, dtype=tf.float64)

        qnode1 = QNode(f, dev_gpumpi, interface="tf", diff_method="adjoint")
        qnode2 = QNode(f, dev_gpumpi, interface="tf", diff_method="parameter-shift")

        with tf.GradientTape() as tape:
            res1 = qnode1(params1, params2)

        g1 = tape.gradient(res1, [params1, params2])

        with tf.GradientTape() as tape:
            res2 = qnode2(params1, params2)

        g2 = tape.gradient(res2, [params1, params2])

        assert np.allclose(g1, g2)

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_interface_torch(self, isBatch_obs):
        """Test if gradients agree between the adjoint and parameter-shift methods when using the
        Torch interface"""
        torch = pytest.importorskip("torch")

        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * torch.sqrt(params2), wires=[0])
            qml.RY(torch.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = torch.tensor(0.3, requires_grad=True)
        params2 = torch.tensor(0.4, requires_grad=True)

        qnode1 = QNode(f, dev_gpumpi, interface="torch", diff_method="adjoint")
        qnode2 = QNode(f, dev_gpumpi, interface="torch", diff_method="parameter-shift")

        res1 = qnode1(params1, params2)
        res1.backward()

        grad_adjoint = params1.grad, params2.grad

        res2 = qnode2(params1, params2)
        res2.backward()

        grad_ps = params1.grad, params2.grad

        assert np.allclose(grad_adjoint, grad_ps, atol=1e-7)

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_interface_jax(self, isBatch_obs):
        """Test if the gradients agree between adjoint and parameter-shift methods in the
        jax interface"""
        jax = pytest.importorskip("jax")

        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * jax.numpy.sqrt(params2), wires=[0])
            qml.RY(jax.numpy.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = jax.numpy.array(0.3)
        params2 = jax.numpy.array(0.4)

        qnode_adjoint = QNode(f, dev_gpumpi, interface="jax", diff_method="adjoint")
        qnode_ps = QNode(f, dev_gpumpi, interface="jax", diff_method="parameter-shift")

        grad_adjoint = jax.grad(qnode_adjoint)(params1, params2)
        grad_ps = jax.grad(qnode_ps)(params1, params2)

        assert np.allclose(grad_adjoint, grad_ps, atol=1e-7)


@pytest.fixture(params=[np.complex64, np.complex128])
def test_qchem_expvalcost_correct(request):
    """EvpvalCost with qchem Hamiltonian work corectly"""
    from pennylane import qchem

    symbols = ["Li", "H"]
    geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])
    H, qubits = qchem.molecular_hamiltonian(
        symbols, geometry, active_electrons=2, active_orbitals=5
    )
    active_electrons = 2
    hf_state = qchem.hf_state(active_electrons, qubits)

    dev_lig = qml.device(
        "lightning.gpu",
        wires=range(qubits),
        mpi=True,
        c_dtype=request.params,
    )

    @qml.qnode(dev_lig, diff_method="adjoint")
    def circuit_1(params, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.RX(params[0], wires=0)
        qml.RY(params[0], wires=1)
        qml.RZ(params[0], wires=2)
        qml.Hadamard(wires=1)
        return qml.expval(H)

    params = np.array([0.123], requires_grad=True)
    grads_lig = qml.grad(circuit_1)(params, wires=range(qubits))

    dev_def = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev_def, diff_method="backprop")
    def circuit_2(params, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.RX(params[0], wires=0)
        qml.RY(params[0], wires=1)
        qml.RZ(params[0], wires=2)
        qml.Hadamard(wires=1)
        return qml.expval(H)

    params = np.array([0.123], requires_grad=True)
    grads_def = qml.grad(circuit_2)(params, wires=range(qubits))

    assert np.allclose(grads_lig, grads_def)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qml.StatePrep(unitary_group.rvs(2**6, random_state=0)[0], wires=wires)
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

    qml.adjoint(qml.CRot(params[21], params[22], params[23], wires=[wires[1], wires[2]]))
    qml.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qml.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])


@pytest.fixture(params=[np.complex64, np.complex128])
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
        (0.5 * qml.PauliZ(0) @ qml.PauliZ(2),),
    ],
)
@pytest.mark.parametrize("isBatch_obs", [False, True])
def test_integration(returns, isBatch_obs, request):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    num_wires = 6
    dev_default = qml.device("default.qubit", wires=range(num_wires))
    dev_gpu = qml.device(
        "lightning.gpu",
        wires=range(num_wires),
        mpi=True,
        c_dtype=request.params,
        batch_obs=isBatch_obs,
    )

    def circuit(params):
        circuit_ansatz(params, wires=range(num_wires))
        return qml.math.hstack([qml.expval(r) for r in returns])

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


@pytest.fixture(params=[np.complex64, np.complex128])
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
@pytest.mark.parametrize("isBatch_obs", [False, True])
def test_integration_custom_wires(returns, isBatch_obs, request):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""
    dev_lightning = qml.device("lightning.qubit", wires=custom_wires)
    dev_gpu = qml.device(
        "lightning.gpu", wires=custom_wires, mpi=True, c_dtype=request.params, batch_obs=isBatch_obs
    )

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


@pytest.fixture(scope="session")
def create_xyz_file(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tmp")
    file = directory / "h2.xyz"
    file.write_text("""2\nH2, Unoptimized\nH  1.0 0.0 0.0\nH -1.0 0.0 0.0""")
    yield file


@pytest.mark.slow
@pytest.fixture(params=[np.complex64, np.complex128])
@pytest.mark.parametrize(
    "dev_compare",
    list(it.product(["default.qubit", "lightning.qubit"])),
)
@pytest.mark.parametrize("isBatch_obs", [False, True])
def test_integration_H2_Hamiltonian(create_xyz_file, dev_compare, isBatch_obs, request):
    skipp_condn = pytest.importorskip("openfermionpyscf")
    n_electrons = 2
    np.random.seed(1337)

    str_path = create_xyz_file
    symbols, coordinates = qml.qchem.read_structure(str(str_path), outpath=str(str_path.parent))

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        method="pyscf",
        active_electrons=n_electrons,
        name="h2",
        outpath=str(str_path.parent),
    )
    hf_state = qml.qchem.hf_state(n_electrons, qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, qubits)

    # Choose different batching supports here
    dev = qml.device(
        "lightning.gpu", wires=qubits, mpi=True, c_dtype=request.params, batch_obs=isBatch_obs
    )
    dev_comp = qml.device(dev_compare, wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params, excitations):
        qml.BasisState(hf_state, wires=H.wires)
        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qml.DoubleExcitation(params[i], wires=excitation)
            else:
                qml.SingleExcitation(params[i], wires=excitation)
        return qml.expval(H)

    @qml.qnode(dev_comp, diff_method="parameter-shift")
    def circuit_compare(params, excitations):
        qml.BasisState(hf_state, wires=H.wires)

        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qml.DoubleExcitation(params[i], wires=excitation)
            else:
                qml.SingleExcitation(params[i], wires=excitation)
        return qml.expval(H)

    jac_func = qml.jacobian(circuit)
    jac_func_comp = qml.jacobian(circuit_compare)

    params = qml.numpy.array([0.0] * len(doubles), requires_grad=True)
    jacs = jac_func(params, excitations=doubles)
    jacs_comp = jac_func_comp(params, excitations=doubles)

    assert np.allclose(jacs, jacs_comp)


@pytest.mark.parametrize(
    "returns",
    [
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0]),
        qml.Hermitian(
            np.kron(qml.PauliY.compute_matrix(), qml.PauliZ.compute_matrix()),
            wires=[custom_wires[3], custom_wires[2]],
        ),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0])
        @ qml.PauliZ(custom_wires[2]),
    ],
)
@pytest.mark.parametrize("isBatch_obs", [False, True])
def test_fail_adjoint_Hermitian(returns, isBatch_obs):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_gpu = qml.device(
        "lightning.gpu", wires=custom_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
    )

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns)

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")

    with pytest.raises(
        qml._device.DeviceError,
        match="Observable Hermitian not supported on device",
    ):
        j_gpu = qml.jacobian(qnode_gpu)(params)


@pytest.mark.parametrize(
    "returns",
    [
        0.6
        * qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0])
        @ qml.PauliX(wires=custom_wires[1]),
    ],
)
@pytest.mark.parametrize("isBatch_obs", [False, True])
def test_fail_adjoint_mixed_Hamiltonian_Hermitian(returns, isBatch_obs):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_gpu = qml.device(
        "lightning.gpu", wires=custom_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
    )

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns)

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")

    with pytest.raises((TypeError, ValueError)):
        j_gpu = qml.jacobian(qnode_gpu)(params)


@pytest.mark.parametrize(
    "returns",
    [
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1], [qml.PauliX(wires=custom_wires[0]) @ qml.PauliY(wires=custom_wires[1])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [2.0], [qml.PauliX(wires=custom_wires[2]) @ qml.PauliZ(wires=custom_wires[0])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [2.0], [qml.PauliX(wires=custom_wires[1]) @ qml.PauliZ(wires=custom_wires[2])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [1.1], [qml.PauliX(wires=custom_wires[0]) @ qml.PauliZ(wires=custom_wires[2])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
    ],
)
def test_adjoint_SparseHamiltonian_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    comm = MPI.COMM_WORLD
    dev_gpu = qml.device("lightning.gpu", wires=custom_wires, mpi=True)
    dev_cpu = qml.device("default.qubit", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns)

    if comm.Get_rank() == 0:
        n_params = 30
        np.random.seed(1337)
        params = np.random.rand(n_params)
    else:
        params = None

    params = comm.bcast(params, root=0)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_cpu = qml.QNode(circuit, dev_cpu, diff_method="parameter-shift")

    j_gpu = qml.jacobian(qnode_gpu)(params)
    j_cpu = qml.jacobian(qnode_cpu)(params)

    assert np.allclose(j_cpu, j_gpu)


@pytest.mark.parametrize(
    "returns",
    [
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliZ(1) @ qml.PauliX(0) @ qml.Identity(2) @ qml.PauliX(4) @ qml.Identity(5)],
            ).sparse_matrix(range(6)),
            wires=range(6),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(1) @ qml.PauliZ(0)],
            ).sparse_matrix(range(6)),
            wires=range(6),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(0)],
            ).sparse_matrix(range(6)),
            wires=range(6),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(5)],
            ).sparse_matrix(range(6)),
            wires=range(6),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(0) @ qml.PauliZ(1)],
            ).sparse_matrix(range(6)),
            wires=range(6),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian([2.0], [qml.PauliX(1) @ qml.PauliZ(2)]).sparse_matrix(range(6)),
            wires=range(6),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian([2.0], [qml.PauliX(2) @ qml.PauliZ(4)]).sparse_matrix(range(6)),
            wires=range(6),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian([1.1], [qml.PauliX(2) @ qml.PauliZ(0)]).sparse_matrix(range(6)),
            wires=range(6),
        ),
    ],
)
def test_adjoint_SparseHamiltonian(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    comm = MPI.COMM_WORLD
    dev_gpu = qml.device("lightning.gpu", wires=6, mpi=True)
    dev_cpu = qml.device("default.qubit", wires=6)

    def circuit(params):
        circuit_ansatz(params, wires=range(6))
        return qml.expval(returns)

    if comm.Get_rank() == 0:
        n_params = 30
        np.random.seed(1337)
        params = np.random.rand(n_params)
    else:
        params = None

    params = comm.bcast(params, root=0)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_cpu = qml.QNode(circuit, dev_cpu, diff_method="parameter-shift")

    j_gpu = qml.jacobian(qnode_gpu)(params)
    j_cpu = qml.jacobian(qnode_cpu)(params)

    assert np.allclose(j_cpu, j_gpu)


@pytest.mark.parametrize(
    "obs,obs_type_c64,obs_type_c128",
    [
        (qml.PauliZ(0), NamedObsGPUMPI_C64, NamedObsGPUMPI_C128),
        (qml.PauliZ(0) @ qml.PauliZ(1), TensorProdObsGPUMPI_C64, TensorProdObsGPUMPI_C128),
        (qml.Hadamard(0), NamedObsGPUMPI_C64, NamedObsGPUMPI_C128),
        (qml.Hamiltonian([1], [qml.PauliZ(0)]), HamiltonianGPUMPI_C64, HamiltonianGPUMPI_C128),
        (
            qml.PauliZ(0) @ qml.Hadamard(1) @ (0.1 * (qml.PauliZ(2) + qml.PauliX(3))),
            HamiltonianGPUMPI_C64,
            HamiltonianGPUMPI_C128,
        ),
        (
            qml.SparseHamiltonian(qml.Hamiltonian([1], [qml.PauliZ(0)]).sparse_matrix(), wires=[0]),
            SparseHamiltonianGPUMPI_C64,
            SparseHamiltonianGPUMPI_C128,
        ),
    ],
)
@pytest.mark.parametrize("use_csingle", [True, False])
@pytest.mark.parametrize("use_mpi", [True])
def test_obs_returns_expected_type(obs, obs_type_c64, obs_type_c128, use_csingle, use_mpi):
    """Tests that observables get serialized to the expected type."""
    obs_type = obs_type_c64 if use_csingle else obs_type_c128
    assert isinstance(
        _serialize_ob(obs, dict(enumerate(obs.wires)), use_csingle, use_mpi, False), obs_type
    )


@pytest.mark.parametrize(
    "bad_obs",
    [
        qml.Hermitian(np.eye(2), wires=0),
        qml.sum(qml.PauliZ(0), qml.Hadamard(1)),
        qml.Projector([0], wires=0),
        qml.PauliZ(0) @ qml.Projector([0], wires=1),
        qml.sum(qml.Hadamard(0), qml.PauliX(1)),
    ],
)
@pytest.mark.parametrize("use_csingle", [True, False])
@pytest.mark.parametrize("use_mpi", [True])
def test_obs_not_supported_for_adjoint_diff(bad_obs, use_csingle, use_mpi):
    """Tests observables that can't be serialized for adjoint-differentiation."""
    with pytest.raises(TypeError, match="Please use Pauli-words only."):
        _serialize_ob(bad_obs, dict(enumerate(bad_obs.wires)), use_csingle, use_mpi)
