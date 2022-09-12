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
Unit tests for the Hamiltonian expval(H) of the :mod:`pennylane_lightning_gpu.LightningGPU` device.
"""
import pytest
import numpy as np
import pennylane as qml
import math

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
    @pytest.fixture(params=[np.complex128])
    def gpu_dev(self, request):
        return LightningGPU(wires=2, c_dtype=request.param)

    @pytest.mark.parametrize(
        "obs, coeffs, res",
        [
            ([qml.PauliX(0) @ qml.PauliZ(1)], [1.0], 0.0),
            ([qml.PauliZ(0) @ qml.PauliZ(1)], [1.0], math.cos(0.4) * math.cos(-0.2)),
            (
                [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)],
                [1.0, 0.2],
                0.2 * math.cos(0.4) * math.cos(-0.2),
            ),
        ],
    )
    def test_expval_hamiltonian(self, obs, coeffs, res, tol, gpu_dev):
        """Test expval with Hamiltonian"""
        ham = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(gpu_dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(ham)

        assert np.allclose(circuit(), res, atol=tol, rtol=0)

    def test_hamiltonan_expectation(self, qubit_device_3_wires, tol):

        dev = qubit_device_3_wires

        obs = qml.PauliX(1) @ qml.PauliY(2)

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
            dtype=np.complex128,
        )

        dev.syncH2D()

        res = dev.expval(H)
        expected = 1

        assert np.allclose(res, expected)
