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
r"""
Helper functions for serializing quantum tapes.
"""
from typing import List, Tuple

import numpy as np
from pennylane import (
    BasisState,
    Hadamard,
    Projector,
    QubitStateVector,
    Rot,
)
from pennylane.grouping import is_pauli_word
from pennylane.operation import Observable, Tensor
from pennylane.ops.qubit.observables import Hermitian
from pennylane.tape import QuantumTape

# Remove after the next release of PL
# Add from pennylane import matrix
import pennylane as qml

try:
    from pennylane_lightning_gpu.lightning_gpu_qubit_ops import (
        LightningGPU_C128,
        LightningGPU_C64,
        NamedObsGPU_C64,
        NamedObsGPU_C128,
        TensorProdObsGPU_C64,
        TensorProdObsGPU_C128,
        HamiltonianGPU_C64,
        HamiltonianGPU_C128,
        SparseHamiltonianGPU_C64,
        SparseHamiltonianGPU_C128,
        HermitianObsGPU_C64,
        HermitianObsGPU_C128,
    )
except ImportError as e:
    print(e)


def _obs_has_kernel(obs: Observable) -> bool:
    """Returns True if the input observable has a supported kernel in the C++ backend.

    Args:
        obs (Observable): the input observable

    Returns:
        bool: indicating whether ``obs`` has a dedicated kernel in the backend
    """
    if is_pauli_word(obs):
        return True
    if isinstance(obs, (Hadamard, Projector)):
        return True
    if isinstance(obs, Tensor):
        return all(_obs_has_kernel(o) for o in obs.obs)
    return False


def _serialize_named_ob(o, wires_map: dict, use_csingle: bool):
    """Serializes an observable (Named)"""
    assert not isinstance(o, Tensor)

    if use_csingle:
        ctype = np.complex64
        named_obs = NamedObsGPU_C64
    else:
        ctype = np.complex128
        named_obs = NamedObsGPU_C128

    wires_list = o.wires.tolist()
    wires = [wires_map[w] for w in wires_list]
    if _obs_has_kernel(o):
        return named_obs(o.name, wires)


def _serialize_tensor_ob(ob, wires_map: dict, use_csingle: bool):
    """Serialize a tensor observable"""
    assert isinstance(ob, Tensor)

    if use_csingle:
        tensor_obs = TensorProdObsGPU_C64
    else:
        tensor_obs = TensorProdObsGPU_C128
    return tensor_obs([_serialize_ob(o, wires_map, use_csingle) for o in ob.obs])


def _serialize_hamiltonian(ob, wires_map: dict, use_csingle: bool):
    if use_csingle:
        rtype = np.float32
        hamiltonian_obs = HamiltonianGPU_C64
    else:
        rtype = np.float64
        hamiltonian_obs = HamiltonianGPU_C128

    coeffs = np.array(ob.coeffs).astype(rtype)
    terms = [_serialize_ob(t, wires_map, use_csingle) for t in ob.ops]
    return [hamiltonian_obs([c], [t]) for (c, t) in zip(coeffs, terms)]


def _serialize_sparsehamiltonian(ob, wires_map: dict, use_csingle: bool):
    if use_csingle:
        ctype = np.complex64
        rtype = np.int32
        sparsehamiltonian_obs = SparseHamiltonianGPU_C64
    else:
        ctype = np.complex128
        rtype = np.int64
        sparsehamiltonian_obs = SparseHamiltonianGPU_C128

    spm = ob.sparse_matrix()
    data = np.array(spm.data).astype(ctype)
    indices = np.array(spm.indices).astype(rtype)
    offsets = np.array(spm.indptr).astype(rtype)

    wires = []
    wires_list = ob.wires.tolist()
    wires.extend([wires_map[w] for w in wires_list])

    return sparsehamiltonian_obs(data, indices, offsets, wires)


def _serialize_hermitian(ob, wires_map: dict, use_csingle: bool):
    if use_csingle:
        rtype = np.float32
        hermitian_obs = HermitianObsGPU_C64
    else:
        rtype = np.float64
        hermitian_obs = HermitianObsGPU_C128

    data = qml.matrix(ob).astype(rtype).ravel(order="C")
    return hermitian_obs(data, ob.wires.tolist())


def _serialize_ob(ob, wires_map, use_csingle):
    if isinstance(ob, Tensor):
        return _serialize_tensor_ob(ob, wires_map, use_csingle)
    elif ob.name == "Hamiltonian":
        return _serialize_hamiltonian(ob, wires_map, use_csingle)
    elif ob.name == "SparseHamiltonian":
        return _serialize_sparsehamiltonian(ob, wires_map, use_csingle)
    elif ob.name == "Hermitian":
        raise TypeError(
            f"Hermitian observables are not currently supported for adjoint differentiation. Please use Pauli-words only."
        )
    else:
        return _serialize_named_ob(ob, wires_map, use_csingle)


def _serialize_observables(tape: QuantumTape, wires_map: dict, use_csingle: bool = False) -> List:
    """Serializes the observables of an input tape.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        list(ObservableGPU_C64 or ObservableGPU_C128): A list of observable objects compatible with the C++ backend
    """
    output = []
    offsets = [0]

    for ob in tape.observables:
        ser_ob = _serialize_ob(ob, wires_map, use_csingle)
        if isinstance(ser_ob, list):
            output.extend(ser_ob)
            offsets.append(offsets[-1] + len(ser_ob))
        else:
            output.append(ser_ob)
            offsets.append(offsets[-1] + 1)
    return output, offsets


def _serialize_ops(
    tape: QuantumTape, wires_map: dict, use_csingle: bool = False
) -> Tuple[List[List[str]], List[np.ndarray], List[List[int]], List[bool], List[np.ndarray]]:
    """Serializes the operations of an input tape.

    The state preparation operations are not included.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        Tuple[list, list, list, list, list]: A serialization of the operations, containing a list
        of operation names, a list of operation parameters, a list of observable wires, a list of
        inverses, and a list of matrices for the operations that do not have a dedicated kernel.
    """
    names = []
    params = []
    wires = []
    inverses = []
    mats = []

    uses_stateprep = False

    sv_py = LightningGPU_C64 if use_csingle else LightningGPU_C128

    for o in tape.operations:
        if isinstance(o, (BasisState, QubitStateVector)):
            uses_stateprep = True
            continue
        elif isinstance(o, Rot):
            op_list = o.expand().operations
        else:
            op_list = [o]

        for single_op in op_list:
            is_inverse = single_op.inverse

            name = single_op.name if not is_inverse else single_op.name[:-4]
            names.append(name)

            if getattr(sv_py, name, None) is None:
                params.append([])
                mats.append(qml.matrix(single_op))

                if is_inverse:
                    is_inverse = False
            else:
                params.append(single_op.parameters)
                mats.append([])

            wires_list = single_op.wires.tolist()
            wires.append([wires_map[w] for w in wires_list])
            inverses.append(is_inverse)

    return (names, params, wires, inverses, mats), uses_stateprep
