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
Unit tests for the SparseHamiltonian expval(H) method of the :mod:`pennylane_lightning_gpu.LightningGPU` device.
"""
import pytest

import numpy as np
import pennylane as qml

from pennylane_lightning_gpu import LightningGPU

try:
    from pennylane_lightning_gpu.lightning_gpu import CPP_BINARY_AVAILABLE

    if not CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU is unsupported on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU is unsupported on this platform. Skipping.",
        allow_module_level=True,
    )


class TestHamiltonianExpval:
    def test_hamiltionan_expectation(self, qubit_device_3_wires, tol):

        dev = qubit_device_3_wires
        obs = qml.Identity(0) @ qml.PauliX(1) @ qml.PauliY(2)

        obs1 = qml.Identity(1)

        H = qml.Hamiltonian([1.0, 1.0], [obs1, obs])

        dev._state = np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.1j,
                0.1 + 0.1j,
                0.1 + 0.2j,
                0.2 + 0.2j,
                0.3 + 0.3j,
                0.3 + 0.4j,
                0.4 + 0.5j,
            ],
            dtype=np.complex64,
        )

        dev.syncH2D()
        Hmat = qml.utils.sparse_hamiltonian(H)
        H_sparse = qml.SparseHamiltonian(Hmat, wires=3)

        res = dev.expval(H_sparse)
        expected = 1

        assert np.allclose(res, expected)


class TestSparseExpval:
    """Tests for the expval function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningGPU(wires=2, c_dtype=request.param)
        # return qml.device("lightning.qubit", wires=2, c_dtype=request.param)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0) @ qml.Identity(1), 0.00000000000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0) @ qml.Identity(1), -0.38941834230865050],
            [qml.Identity(0) @ qml.PauliY(1), 0.00000000000000000],
            [qml.PauliZ(0) @ qml.Identity(1), 0.92106099400288520],
            [qml.Identity(0) @ qml.PauliZ(1), 0.98006657784124170],
        ],
    )
    def test_sparse_Pauli_words(self, cases, tol, dev):
        """Test expval of some simple sparse Hamiltonian"""

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.utils.sparse_hamiltonian(qml.Hamiltonian([1], [cases[0]])), wires=[0, 1]
                )
            )

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0) @ qml.Identity(1), 0.00000000000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0) @ qml.Identity(1), -0.38941834230865050],
            [qml.Identity(0) @ qml.PauliY(1), 0.00000000000000000],
            [qml.Hermitian([[1, 0], [0, -1]], wires=0) @ qml.Identity(1), 0.92106099400288520],
            [qml.Identity(0) @ qml.PauliZ(1), 0.98006657784124170],
            [
                qml.Hermitian([[1, 0], [0, 1]], wires=0)
                @ qml.Hermitian([[1, 0], [0, -1]], wires=1),
                0.98006657784124170,
            ],
        ],
    )
    def test_sparse_arbitrary(self, cases, tol, dev):
        """Test expval of some simple sparse Hamiltonian"""

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.utils.sparse_hamiltonian(qml.Hamiltonian([1], [cases[0]])), wires=[0, 1]
                )
            )

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)
