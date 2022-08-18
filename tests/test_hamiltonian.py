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
Unit tests for the var method of the :mod:`pennylane_lightning_gpu.LightningGPU` device.
"""
import pytest

import numpy as np
import pennylane as qml

try:
    from pennylane_lightning_gpu.lightning_gpu import CPP_BINARY_AVAILABLE

    if not CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU is unsupported on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU is unsupported on this platform. Skipping.",
        allow_module_level=True,
    )

np.random.seed(42)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestHamiltonianExpval:
    def test_hamiltionian_expectation(self, theta, phi, varphi, qubit_device_3_wires, tol):

        dev = qubit_device_3_wires

        #obs = qml.PauliX(0) @ qml.PauliX(1)@ qml.PauliZ(2)
        
        #obs = qml.Identity(0)@ qml.PauliZ(1) @ qml.PauliX(2)

        obs1 = qml.Identity(1)

        #obs2 = qml.PauliX(2) @ qml.Identity(0) @ qml.PauliY(1)

        obs = qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2)

        H = qml.Hamiltonian(
            [1.0,1.0], [obs, obs1]
            #[1.0,1.0,-1.0 ], [qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliX(2), qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2), qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)]
        )

        dev._state = np.array([0.0 + 0.0j, 0.0 + 0.1j, 0.1 + 0.1j, 0.1 + 0.2j, 0.2 + 0.2j, 0.3 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j])
        print(dev._state)
        
        dev.apply(
            [
                qml.Identity(wires=[1]),
                #qml.RX(theta, wires=[0]),
                #qml.RY(phi, wires=[1]),
                #qml.RZ(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            rotations=obs.diagonalizing_gates(),
        )

        res = dev.expval(H)

        print(dev._state)
        #wires_list = []
        #for i in range(len(H.ops)):
        #    wires_list.append([])
        #    for j in range(len(H.ops[i].wires)):
        #        wires_list[i].append(H.ops[i].wires[j])

        # print(wires_list)

        #wire_list = [
        #    [H.ops[i].wires[j] for j in range(len(H.ops[i].wires))] for i in range(len(H.ops))
        #]

        #print(wire_list, type(wire_list))

        #supported_pauli_basis = ["PauliX", "PauliY", "PauliZ", "Identity"]

        #obs_supported = False

        # name_list = [[if len(H.ops[i].wires) == 1: H.ops[i].name[j] else: H.ops[i].name for j in range(len(H.ops[i].wires))] for i in range(len(H.ops))]

        #name_list = []
        # print(len(H.ops))

        #for i in range(len(H.ops)):
        #    name_list.append([])
        #    for j in range(len(H.ops[i].wires)):
        #        obst = H.ops[i].name[j]
        #        if len(H.ops[i].wires) == 1:
        #            obst = H.ops[i].name
        #        name_list[i].append(obst)
        #        if obst not in supported_pauli_basis:
        #            obs_supported = False
        #            break
        #        else:
        #            obs_supported = True
        #    else:
        #        continue
        #    break

        # print(obs_supported)

        #print(name_list, len(name_list[0]), len(name_list[1]), len(name_list[2]))
        # print(type(H.ops[0].name), len(H.ops[1].name))

        #print(H.coeffs, type(H.coeffs))

        # print(type(H.ops[0].name[0]),len(H.ops[0].name))
        # print(type(H.ops),H.ops[0].wires,H.ops[1].wires,H.ops[2].wires)
        # print(len(np.asarray(obs.name)),type(obs.wires))
        expected0 = dev.expval(obs) + dev.expval(obs1)# + dev.expval(obs2)
        expected =  np.cos(varphi) * np.cos(phi)
        assert np.allclose(res, expected0)
        #assert np.allclose(expected0, expected)
