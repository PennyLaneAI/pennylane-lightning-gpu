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
Tests for the ``vjp`` method of`pennylane_lightning_gpu.LightningGPU`.
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


class TestVectorJacobianProduct:
    """Tests for the `vjp` function"""

    @pytest.fixture(params=[np.complex128])
    def dev(self, request):
        return LightningGPU(wires=3, c_dtype=request.param)

    def test_multiple_measurements(self, tol, dev):
        """Tests provides correct answer when provided multiple measurements."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliX(0))
            qml.expval(qml.PauliY(1))
            qml.expval(qml.PauliZ(1))

        dy = np.array([1.0, 2.0, 3.0])
        tape1.trainable_params = {1, 2, 3}

        with qml.tape.QuantumTape() as tape2:
            ham = qml.Hamiltonian(dy, [qml.PauliX(0), qml.PauliY(1), qml.PauliY(1)])
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(ham)

        tape2.trainable_params = {1, 2, 3}

        fn1 = dev.vjp(tape1.measurements, dy, use_device_state=True)

        vjp1 = fn1(tape1)

        vjp2 = dev.adjoint_jacobian(tape2, use_device_state=True)

        assert np.allclose(vjp1, vjp2.ravel(order="C"), atol=tol, rtol=0)
