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
This module contains the :class:`~.LightningGPU` class, a PennyLane simulator device that
interfaces with the NVIDIA cuQuantum cuStateVec simulator library for GPU-enabled calculations.
"""
from warnings import warn

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from pennylane import (
    math,
    BasisState,
    QubitStateVector,
    DeviceError,
    Projector,
    Hermitian,
    Rot,
    QuantumFunctionError,
    QubitStateVector,
)
from pennylane_lightning import LightningQubit
from pennylane.operation import Tensor, Operation
from pennylane.measurements import Expectation
from pennylane.wires import Wires

# Remove after the next release of PL
# Add from pennylane import matrix
import pennylane as qml

from ._version import __version__

try:
    from .lightning_gpu_qubit_ops import (
        LightningGPU_C128,
        LightningGPU_C64,
        AdjointJacobianGPU_C128,
        AdjointJacobianGPU_C64,
        device_reset,
        is_gpu_supported,
        get_gpu_arch,
        DevPool,
        DevTag,
        ObsStructGPU_C128,
        ObsStructGPU_C64,
        OpsStructGPU_C128,
        OpsStructGPU_C64,
    )

    from ._serialize import _serialize_obs, _serialize_ops
    from ctypes.util import find_library
    from importlib import util as imp_util

    if find_library("custatevec") == None and not imp_util.find_spec("cuquantum"):
        raise ImportError(
            'cuQuantum libraries not found. Please check your "LD_LIBRARY_PATH" environment variable,'
            'or ensure you have installed the appropriate distributable "cuQuantum" package.'
        )
    if not is_gpu_supported():
        raise ValueError(f"CUDA device is an unsupported version: {get_gpu_arch()}")

    CPP_BINARY_AVAILABLE = True
except (ModuleNotFoundError, ImportError, ValueError) as e:
    warn(str(e), UserWarning)
    CPP_BINARY_AVAILABLE = False


def _gpu_dtype(dtype):
    if dtype not in [np.complex128, np.complex64]:
        raise ValueError(f"Data type is not supported for state-vector computation: {dtype}")
    return LightningGPU_C128 if dtype == np.complex128 else LightningGPU_C64


class LightningGPU(LightningQubit):
    """PennyLane-Lightning-GPU device.

    Args:
        wires (int): the number of wires to initialize the device with
        sync (bool): immediately sync with host-sv after applying operations
        c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
    """

    name = "PennyLane plugin for GPU-backed Lightning device using NVIDIA cuQuantum SDK"
    short_name = "lightning.gpu"
    pennylane_requires = ">=0.22"
    version = __version__
    author = "Xanadu Inc."
    _CPP_BINARY_AVAILABLE = True

    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Identity",
    }

    def __init__(self, wires, *, sync=True, c_dtype=np.complex128, shots=None, batch_obs=False):
        super().__init__(wires, c_dtype=c_dtype, shots=shots)
        self._gpu_state = _gpu_dtype(self._state.dtype)(self._state)
        self._sync = sync
        self._dp = DevPool()
        self._batch_obs = batch_obs

    def reset(self):
        super().reset()
        self._gpu_state.resetGPU(False)  # Sync reset

    def syncH2D(self, use_async=False):
        """Explicitly synchronize CPU data to GPU"""
        self._gpu_state.HostToDevice(self._state.ravel(order="C"), use_async)

    def syncD2H(self, use_async=False):
        """Explicitly synchronize GPU data to CPU"""
        self._gpu_state.DeviceToHost(self._state.ravel(order="C"), use_async)
        self._pre_rotated_state = self._state

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_reversible_diff=False,
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            supports_finite_shots=True,
            returns_state=True,
        )
        capabilities.pop("passthru_devices", None)
        return capabilities

    def statistics(self, observables, shot_range=None, bin_size=None):
        ## Ensure D2H sync before calculating non-GPU supported operations
        if self._sync:
            self.syncD2H()
        return super().statistics(observables, shot_range, bin_size)

    def apply_cq(self, operations, **kwargs):
        # Skip over identity operations instead of performing
        # matrix multiplication with the identity.
        skipped_ops = ["Identity"]

        for o in operations:
            if o.base_name in skipped_ops:
                continue
            name = o.name.split(".")[0]  # The split is because inverse gates have .inv appended
            method = getattr(self._gpu_state, name, None)

            wires = self.wires.indices(o.wires)

            if method is None:
                # Inverse can be set to False since qml.matrix(o) is already in inverted form
                try:
                    mat = qml.matrix(o)
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    mat = o.matrix

                if len(mat) == 0:
                    raise Exception("Unsupported operation")
                self._gpu_state.apply(
                    name,
                    wires,
                    False,
                    [],
                    mat.ravel(order="C"),  # inv = False: Matrix already in correct form;
                )  # Parameters can be ignored for explicit matrices; F-order for cuQuantum

            else:
                inv = o.inverse
                param = o.parameters
                method(wires, inv, param)

    def apply(self, operations, **kwargs):
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], QubitStateVector):
                self._apply_state_vector(operations[0].parameters[0].copy(), operations[0].wires)
                del operations[0]
                self.syncH2D()
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                del operations[0]
                self.syncH2D()

        for operation in operations:
            if isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been "
                    "applied on a {} device.".format(operation.name, self.short_name)
                )

        self.apply_cq(operations)
        if self._sync:
            self.syncD2H()

    def adjoint_diff_support_check(self, tape):
        """Check Lightning adjoint differentiation method support for a tape.

        Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
        observables, or operations by the Lightning adjoint differentiation method.

        Args:
            tape (.QuantumTape): quantum tape to differentiate
        """
        for m in tape.measurements:
            if m.return_type is not Expectation:
                raise QuantumFunctionError(
                    "Adjoint differentiation method does not support"
                    f" measurement {m.return_type.value}"
                )
            if not isinstance(m.obs, Tensor):
                if isinstance(m.obs, Projector):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
                if isinstance(m.obs, Hermitian):
                    raise QuantumFunctionError(
                        "Lightning adjoint differentiation method does not currently support the Hermitian observable"
                    )
            else:
                if any([isinstance(o, Projector) for o in m.obs.non_identity_obs]):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
                if any([isinstance(o, Hermitian) for o in m.obs.non_identity_obs]):
                    raise QuantumFunctionError(
                        "Lightning adjoint differentiation method does not currently support the Hermitian observable"
                    )

        for op in tape.operations:
            if op.num_params > 1 and not isinstance(op, Rot):
                raise QuantumFunctionError(
                    f"The {op.name} operation is not supported using "
                    'the "adjoint" differentiation method'
                )

    def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False, **kwargs):
        if self.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " The derivative is always exact when using the adjoint differentiation method.",
                UserWarning,
            )

        if len(tape.trainable_params) == 0:
            return np.array(0)

        # Check adjoint diff support
        self.adjoint_diff_support_check(tape)

        # Initialization of state
        if starting_state is not None:
            ket = np.ravel(starting_state, order="C")
        else:
            if not use_device_state:
                self.reset()
                self.execute(tape)
            ket = np.ravel(self._pre_rotated_state, order="C")

        if self.use_csingle:
            adj = AdjointJacobianGPU_C64()
            ket = ket.astype(np.complex64)
        else:
            adj = AdjointJacobianGPU_C128()

        obs_serialized = _serialize_obs(tape, self.wire_map, use_csingle=self.use_csingle)
        ops_serialized, use_sp = _serialize_ops(tape, self.wire_map, use_csingle=self.use_csingle)

        ops_serialized = adj.create_ops_list(*ops_serialized)

        trainable_params = sorted(tape.trainable_params)

        tp_shift = []
        record_tp_rows = []
        all_params = 0

        for op_idx, tp in enumerate(trainable_params):
            op, _ = tape.get_operation(
                op_idx
            )  # get op_idx-th operator among differentiable operators

            if isinstance(op, Operation) and not isinstance(op, (BasisState, QubitStateVector)):
                # We now just ignore non-op or state preps
                tp_shift.append(tp)
                record_tp_rows.append(all_params)
            all_params += 1

        if use_sp:
            # When the first element of the tape is state preparation. Still, I am not sure
            # whether there must be only one state preparation...
            tp_shift = [i - 1 for i in tp_shift]

        if self._dp.getTotalDevices() > 1 and self._batch_obs:
            jac = adj.adjoint_jacobian_batched(
                self._gpu_state,
                obs_serialized,
                ops_serialized,
                tp_shift,
            )
        else:
            jac = adj.adjoint_jacobian(self._gpu_state, obs_serialized, ops_serialized, tp_shift)

        jac = np.array(jac)  # only for parameters differentiable with the adjoint method
        jac = jac.reshape(-1, len(tp_shift))
        jac_r = np.zeros((jac.shape[0], all_params))
        jac_r[:, record_tp_rows] = jac
        return jac_r

    def sample(self, observable, shot_range=None, bin_size=None, counts=False):
        if observable.name != "PauliZ":
            self.apply_cq(observable.diagonalizing_gates())
            self._samples = self.generate_samples()
        return super().sample(observable, shot_range=shot_range, bin_size=bin_size, counts=counts)

    def expval(self, observable, shot_range=None, bin_size=None):
        if observable.name in [
            "Projector",
            "Hamiltonian",
            "SparseHamiltonian",
        ]:
            self.syncD2H()
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        if self.shots is not None:
            # estimate the expectation value
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.mean(samples, axis=0))

        par = (
            observable.parameters
            if (
                len(observable.parameters) > 0 and isinstance(observable.parameters[0], np.floating)
            )
            else []
        )
        return self._gpu_state.ExpectationValue(
            observable.name,
            self.wires.indices(observable.wires),
            par,  # observables should not pass parameters, use matrix instead
            qml.matrix(observable).ravel(order="C"),
        )

    def probability(self, wires=None, shot_range=None, bin_size=None):
        if self.shots is not None:
            return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)

        wires = wires or self.wires
        wires = Wires(wires)

        # translate to wire labels used by device
        device_wires = self.map_wires(wires)
        # Device returns as col-major orderings, so perform transpose on data for bit-index shuffle for now.
        return (
            self._gpu_state.Probability(device_wires)
            .reshape([2] * len(wires))
            .transpose()
            .reshape(-1)
        )

    def generate_samples(self):
        """Generate samples

        Returns:
            array[int]: array of samples in binary representation with shape ``(dev.shots, dev.num_wires)``
        """
        return self._gpu_state.GenerateSamples(len(self.wires), self.shots).astype(int)

    def var(self, observable, shot_range=None, bin_size=None):
        if self.shots is not None:
            # estimate the var
            # Lightning doesn't support sampling yet
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.var(samples, axis=0))

        adjoint_matrix = math.T(math.conj(qml.matrix(observable)))
        sqr_matrix = np.matmul(adjoint_matrix, qml.matrix(observable))

        mean = self._gpu_state.ExpectationValue(
            [i + "_var" for i in observable.name],
            self.wires.indices(observable.wires),
            observable.parameters,
            qml.matrix(observable).ravel(order="C"),
        )

        squared_mean = self._gpu_state.ExpectationValue(
            [i + "_sqr" for i in observable.name],
            self.wires.indices(observable.wires),
            observable.parameters,
            sqr_matrix.ravel(order="C"),
        )

        return squared_mean - (mean**2)


if not CPP_BINARY_AVAILABLE:

    class LightningGPU(LightningQubit):
        name = "PennyLane plugin for GPU-backed Lightning device using NVIDIA cuQuantum SDK: Lightning CPU fall-back"
        short_name = "lightning.gpu"
        pennylane_requires = ">=0.22"
        version = __version__
        author = "Xanadu Inc."
        _CPP_BINARY_AVAILABLE = False

        def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
            w_msg = """
            !!!#####################################################################################
            !!!
            !!! WARNING: INSUFFICIENT SUPPORT DETECTED FOR GPU DEVICE WITH `lightning.gpu`
            !!!          DEFAULTING TO CPU DEVICE `lightning.qubit`
            !!!
            !!!#####################################################################################
            """
            warn(
                w_msg,
                RuntimeWarning,
            )
            super().__init__(wires, c_dtype=c_dtype, **kwargs)
