# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that a Lightning-GPU device has the right attributes, arguments and methods."""
# pylint: disable=no-self-use
import pytest
import numpy as np
import pennylane as qml
import math

from pennylane_lightning_gpu import LightningGPU

try:
    from pennylane_lightning_gpu.lightning_gpu import CPP_BINARY_AVAILABLE
    import pennylane_lightning_gpu as plg

    if not CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU is unsupported on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU is unsupported on this platform. Skipping.",
        allow_module_level=True,
    )


class TestState:
    """Tests for the state() method."""

    test_data_no_parameters = [
        (qml.PauliX, [1, 0], np.array([0, 1])),
        (
            qml.PauliX,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
        ),
        (qml.PauliY, [1, 0], [0, 1j]),
        (
            qml.PauliY,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [-1j / math.sqrt(2), 1j / math.sqrt(2)],
        ),
        (qml.PauliZ, [1, 0], [1, 0]),
        (
            qml.PauliZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), -1 / math.sqrt(2)],
        ),
        (qml.S, [1, 0], [1, 0]),
        (
            qml.S,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 1j / math.sqrt(2)],
        ),
        (qml.T, [1, 0], [1, 0]),
        (
            qml.T,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), np.exp(1j * np.pi / 4) / math.sqrt(2)],
        ),
        (qml.Hadamard, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        (qml.Hadamard, [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
    def test_apply_operation_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        dev = qubit_device_1_wire
        gpu_ctor = plg.lightning_gpu._gpu_dtype(dev.C_DTYPE)
        dev._gpu_state = gpu_ctor(np.array(input).astype(dev.C_DTYPE))
        dev.apply([operation(wires=[0])])

        state_vector = dev.state()
        assert np.allclose(state_vector, np.array(expected_output), atol=tol, rtol=0)
