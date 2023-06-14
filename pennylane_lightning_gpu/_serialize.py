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
    PauliX,
    PauliY,
    PauliZ,
    Identity,
    QubitStateVector,
    Rot,
)
from pennylane.operation import Tensor
from pennylane.ops.op_math import Adjoint
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

    try:
        from .lightning_gpu_qubit_ops import (
            LightningGPUMPI_C128,
            LightningGPUMPI_C64,
            AdjointJacobianGPUMPI_C128,
            AdjointJacobianGPUMPI_C64,
            MPIManager,
            NamedObsGPUMPI_C64,
            NamedObsGPUMPI_C128,
            TensorProdObsGPUMPI_C64,
            TensorProdObsGPUMPI_C128,
            HamiltonianGPUMPI_C64,
            HamiltonianGPUMPI_C128,
            HermitianObsGPUMPI_C64,
            HermitianObsGPUMPI_C128,
        )

        MPI_SUPPORT = True
    except:
        MPI_SUPPORT = False

except ImportError as e:
    print(e)

pauli_name_map = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


def _named_ob_dtype(use_csingle, use_mpi: bool):
    if not use_mpi:
        return NamedObsGPU_C64 if use_csingle else NamedObsGPU_C128
    return NamedObsGPUMPI_C64 if use_csingle else NamedObsGPUMPI_C128


def _tensor_ob_dtype(use_csingle, use_mpi: bool):
    if not use_mpi:
        return TensorProdObsGPU_C64 if use_csingle else TensorProdObsGPU_C128
    return TensorProdObsGPUMPI_C64 if use_csingle else TensorProdObsGPUMPI_C128


def _hermitian_ob_dtype(use_csingle, use_mpi: bool):
    if not use_mpi:
        return (
            [HermitianObsGPU_C64, np.float32] if use_csingle else [HermitianObsGPU_C128, np.float64]
        )
    return (
        [HermitianObsGPUMPI_C64, np.float32]
        if use_csingle
        else [HermitianObsGPUMPI_C128, np.float64]
    )


def _hamiltonian_ob_dtype(use_csingle, use_mpi: bool):
    if not use_mpi:
        return (
            [HamiltonianGPU_C64, np.float32] if use_csingle else [HamiltonianGPU_C128, np.float64]
        )
    return (
        [HamiltonianGPUMPI_C64, np.float32] if use_csingle else [HamiltonianGPUMPI_C128, np.float64]
    )


def _sv_py_dtype(use_csingle, use_mpi: bool):
    if not use_mpi:
        return LightningGPU_C64 if use_csingle else LightningGPU_C128
    return LightningGPUMPI_C64 if use_csingle else LightningGPUMPI_C128


def _serialize_named_ob(o, wires_map: dict, use_csingle: bool, use_mpi: bool):
    """Serializes an observable (Named)"""
    named_obs = _named_ob_dtype(use_csingle, use_mpi)
    wires = [wires_map[w] for w in o.wires]
    if o.name == "Identity":
        wires = wires[:1]
    return named_obs(o.name, wires)


def _serialize_tensor_ob(ob, wires_map: dict, use_csingle: bool, use_mpi: bool):
    """Serialize a tensor observable"""
    assert isinstance(ob, Tensor)
    tensor_obs = _tensor_ob_dtype(use_csingle, use_mpi)
    return tensor_obs([_serialize_ob(o, wires_map, use_csingle, use_mpi) for o in ob.obs])


def _serialize_hamiltonian(
    ob, wires_map: dict, use_csingle: bool, use_mpi: bool, split_terms: bool = True
):
    hamiltonian_obs, rtype = _hamiltonian_ob_dtype(use_csingle, use_mpi)
    coeffs = np.array(ob.coeffs).astype(rtype)
    terms = [_serialize_ob(t, wires_map, use_csingle, use_mpi) for t in ob.ops]

    if split_terms:
        return [hamiltonian_obs([c], [t]) for (c, t) in zip(coeffs, terms)]
    return hamiltonian_obs(coeffs, terms)


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


def _serialize_hermitian(ob, wires_map: dict, use_csingle: bool, use_mpi: bool):
    hermitian_obs, rtype = _hermitian_ob_dtype(use_csingle, use_mpi)

    data = qml.matrix(ob).astype(rtype).ravel(order="C")
    return hermitian_obs(data, ob.wires.tolist())


def _serialize_pauli_word(ob, wires_map: dict, use_csingle: bool, use_mpi: bool):
    """Serialize a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable."""
    named_obs = _named_ob_dtype(use_csingle, use_mpi)
    tensor_obs = _tensor_ob_dtype(use_csingle, use_mpi)

    if len(ob) == 1:
        wire, pauli = list(ob.items())[0]
        return named_obs(pauli_name_map[pauli], [wires_map[wire]])

    return tensor_obs(
        [named_obs(pauli_name_map[pauli], [wires_map[wire]]) for wire, pauli in ob.items()]
    )


def _serialize_pauli_sentence(
    ob, wires_map: dict, use_csingle: bool, use_mpi: bool, split_terms: bool = True
):
    """Serialize a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian."""
    hamiltonian_obs, rtype = _hamiltonian_ob_dtype(use_csingle, use_mpi)

    pwords, coeffs = zip(*ob.items())
    terms = [_serialize_pauli_word(pw, wires_map, use_csingle, use_mpi) for pw in pwords]
    coeffs = np.array(coeffs).astype(rtype)
    if split_terms:
        return [hamiltonian_obs([c], [t]) for (c, t) in zip(coeffs, terms)]
    return hamiltonian_obs(coeffs, terms)


def _serialize_ob(ob, wires_map, use_csingle, use_mpi: bool = False, use_splitting: bool = True):
    if isinstance(ob, Tensor):
        return _serialize_tensor_ob(ob, wires_map, use_csingle, use_mpi)
    elif ob.name == "Hamiltonian":
        return _serialize_hamiltonian(ob, wires_map, use_csingle, use_mpi, use_splitting)
    elif ob.name == "SparseHamiltonian":
        if use_mpi:
            raise TypeError("SparseHamiltonian is not supported for MPI backend.")
        return _serialize_sparsehamiltonian(ob, wires_map, use_csingle)
    elif isinstance(ob, (PauliX, PauliY, PauliZ, Identity, Hadamard)):
        return _serialize_named_ob(ob, wires_map, use_csingle, use_mpi)
    elif ob._pauli_rep is not None:
        return _serialize_pauli_sentence(
            ob._pauli_rep, wires_map, use_csingle, use_mpi, use_splitting
        )
    elif ob.name == "Hermitian":
        raise TypeError(
            "Hermitian observables are not currently supported for adjoint differentiation. Please use Pauli-words only."
        )
    else:
        raise TypeError(f"Unknown observable found: {ob}. Please use Pauli-words only.")


def _serialize_observables(
    tape: QuantumTape, wires_map: dict, use_csingle: bool = False, use_mpi: bool = False
) -> List:
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
        ser_ob = _serialize_ob(ob, wires_map, use_csingle, use_mpi)
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
            is_inverse = isinstance(single_op, Adjoint)

            name = single_op.name if not is_inverse else single_op.base.name
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
