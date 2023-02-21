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
Unit tests for the generate_samples method of the :mod:`pennylane_lightning_gpu.LightningGPU` device.
"""
import pytest

import numpy as np
import pennylane as qml
import pennylane_lightning_gpu as plk

try:
    from pennylane_lightning_gpu import LightningGPU

    if not LightningGPU._CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU is unsupported on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU is unsupported on this platform. Skipping.",
        allow_module_level=True,
    )
np.random.seed(42)


class TestProbs:
    """Test Probs in lightning.gpu"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def test_probs_dtype64(self, request):
        """Test if probs changes the state dtype"""

        dev = qml.device("lightning.gpu", wires=2, c_dtype=request.param)

        dev._state = dev._asarray(
            np.array([1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0]).astype(dev.C_DTYPE)
        )
        p = dev.probability(wires=[0, 1])

        assert dev._state.dtype == dev.C_DTYPE
        assert np.allclose(p, [0.5, 0.5, 0, 0])

    @pytest.fixture(params=[np.complex64, np.complex128])
    def test_probs_H(self, tol, request):
        """Test probs with Hadamard"""
        dev = qml.device("lightning.gpu", wires=2, c_dtype=request.param)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.probs(wires=[0, 1])

        assert np.allclose(circuit(), [0.5, 0.5, 0.0, 0.0], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [None, [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
        ],
    )
    @pytest.mark.xfail
    def test_probs_tape_nowires(self, cases, tol):
        """Test probs with a circuit on wires=[0]"""
        dev = qml.device("lightning.gpu", wires=2, c_dtype=np.complex128)

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [[0, 1], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [0, [0.9165164490394898, 0.08348355096051052]],
            [[0], [0.9165164490394898, 0.08348355096051052]],
        ],
    )
    def test_probs_tape_wire0(self, cases, tol):
        """Test probs with a circuit on wires=[0]"""
        dev = qml.device("lightning.gpu", wires=2, c_dtype=np.complex128)

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [[1, 0], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
            [["a", "0"], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
        ],
    )
    def test_fail_probs_tape_wire0(self, cases, tol):
        """Test probs with a circuit on wires=[0]"""
        dev = qml.device("lightning.gpu", wires=2, c_dtype=np.complex128)

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        with pytest.raises(
            RuntimeError,
            match="Lightning does not currently support out-of-order indices for probabilities",
        ):
            assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [
                [0, 1],
                [
                    0.9178264236525453,
                    0.02096485729264079,
                    0.059841820910257436,
                    0.0013668981445561978,
                ],
            ],
            [0, [0.938791280945186, 0.061208719054813635]],
            [[0], [0.938791280945186, 0.061208719054813635]],
        ],
    )
    def test_probs_tape_wire01(self, cases, tol):
        """Test probs with a circuit on wires=[0,1]"""
        dev = qml.device("lightning.gpu", wires=2, c_dtype=np.complex128)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.5, wires=[0])
            qml.RY(0.3, wires=[1])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [
                [1, 0],
                [
                    0.9178264236525453,
                    0.059841820910257436,
                    0.02096485729264079,
                    0.0013668981445561978,
                ],
            ],
        ],
    )
    def test_fail_probs_tape_wire01(self, cases, tol):
        """Test probs with a circuit on wires=[0,1]"""
        dev = qml.device("lightning.gpu", wires=2, c_dtype=np.complex128)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.5, wires=[0])
            qml.RY(0.3, wires=[1])
            return qml.probs(wires=cases[0])

        with pytest.raises(
            RuntimeError,
            match="Lightning does not currently support out-of-order indices for probabilities",
        ):
            assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)
