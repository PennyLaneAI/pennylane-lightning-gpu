# Release 0.31.0

### New features since last release
 * Add multi-node/multi-GPU support to adjoint methods. 
 [(#119)] (https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/119)
 
 Note each MPI process will return the overall result of the adjoint method. The MPI adjoint method has two options:
 1. The default method is faster if the available problem fits into GPU memory, and will simply enabled with the `mpi=True` device argument. With the default method, a separate `bra` is created for each observable and the `ket` is only updated once for each operation, regardless of the number of observables. This approach may consume more memory due to the up-front creation of multiple `bra`s.  
 2. The memory-optimized method requires less memory but is slower due serialization of the execution. The memory-optimized method uses a single `bra` object that is reused for all observables. The `ket` needs to be updated `n` times, where `n` is the number of observables, for each operation. This approach reduces memory consumption as only one `bra` object is created. However, it may lead to slower execution due to the multiple `ket` updates per gate operation.
 
 Each ``MPI`` process will return the overall simulation results for the adjoint method.

 The workflow for the default adjoint method with MPI support is as follows:
 ```python
  from mpi4py import MPI
  import pennylane as qml
  from pennylane import numpy as np
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  n_wires = 20
  n_layers = 2
  
  dev = qml.device('lightning.gpu', wires= n_wires, mpi=True)
  @qml.qnode(dev, diff_method="adjoint")
  def circuit_adj(weights):
      qml.StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
      return qml.math.hstack([qml.expval(qml.PauliZ(i)) for i in range(n_wires)])
  
  if rank == 0:
      params = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
  else:
      params = None
  
  params = comm.bcast(params, root=0)
  jac = qml.jacobian(circuit_adj)(params)
 ```

 To enable the memory-optimized method, `batch_obs` should be set as `True`. The workflow for the memory-optimized method is as follows:
  ```python
  dev = qml.device('lightning.gpu', wires= n_wires, mpi=True, batch_obs=True)
 ```

 * Add multi-node/multi-GPU support to measurement methods, including `expval`, `generate_samples` and `probability`.
 [(#116)] (https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/116)

 Note that each MPI process will return the overall result of expectation value and sample generation. However, `probability` will 
 return local probability results. Users should be responsible to collect probability results across the MPI processes.
 
 The workflow for collecting probability results across the MPI processes is as follows:
 ```python
 from mpi4py import MPI
 import pennylane as qml
 import numpy as np

 comm = MPI.COMM_WORLD
 rank = comm.Get_rank()
 dev = qml.device('lightning.gpu', wires=8, mpi=True)
 prob_wires = [0, 1]

 @qml.qnode(dev)
 def mpi_circuit():
     qml.Hadamard(wires=1)
     return qml.probs(wires=prob_wires)

 local_probs = mpi_circuit()
 
 #For data collection across MPI processes.
 recv_counts = comm.gather(len(local_probs),root=0)
 if rank == 0:
      probs = np.zeros(2**len(prob_wires))
 else:
      probs = None

 comm.Gatherv(local_probs,[probs,recv_counts],root=0)
 if rank == 0:
    print(probs)
 ```
* Add multi-node/multi-gpu support to gate operation.
  [(#112)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/112)

  This new feature empowers users to leverage the computational power of multi-node and multi-GPUs for running large-scale applications. It requires both the total number of overall `MPI` processes and the number of `MPI` processes of each node to be the same and power of `2`. Each `MPI` process is responsible for managing one GPU for the moment. 
  To enable this feature, users can set `mpi=True`. Furthermore, users can fine-tune the performance of `MPI` operations by adjusting the `mpi_buf_size` parameter. This parameter determines the allocation of `mpi_buf_size` MiB (mebibytes, `2^20` bytes) GPU memory for `MPI` operations. Note that `mpi_buf_size` should be also power of 2 and there will be a runtime warning if GPU memory buffer for MPI operation is larger than the GPU memory allocated for the local state vector. By default (`mpi_buf_size=0`), the GPU memory allocated for MPI operations will be the same of size of the local state vector, with a upper limit of 64 MiB. Note that MiB (`2^20` bytes) is different from MB (megabytes, `10^6` bytes).
  The workflow for the new feature is as follows:
  ```python
  from mpi4py import MPI
  import pennylane as qml
  dev = qml.device('lightning.gpu', wires=8, mpi=True, mpi_buf_size=1)
  @qml.qnode(dev)
  def circuit_mpi():
      qml.PauliX(wires=[0])
      return qml.state()
  local_state_vector = circuit_mpi()
  print(local_state_vector)
  ``` 
  Note that each MPI process will return its local state vector with `qml.state()` here.

### Breaking changes

* Update tests to be compliant with PennyLane v0.31.0 development changes and deprecations.
  [(#114)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/114)

### Improvements

* Use `Operator.name` instead of `Operation.base_name`.
  [(#115)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/115)

* Updated runs-on label for self-hosted runner workflows.
  [(#117)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/117)

* Update workflow to support multi-gpu self-hosted runner.
  [(#118)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/118)

* Add compat workflows.
  [(#121)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/121)

### Documentation

* Update `README.rst` and `CHANGLOG.md` for the MPI backend.
  [(#122)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/122)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Rashid N H M, Shuli Shu

---

# Release 0.30.0

### New features since last release

### Improvements

* Wheels are now checked with `twine check` post-creation for PyPI compatibility.
  [(#103)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/103)

### Bug fixes

* Fix CUDA version to 11 for cuquantum dependency in CI. 
  [(#107)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/107)

* Fix the controlled-gate generators, which are now fully used in the adjoint pipeline following PennyLane PR [(#3874)](https://github.com/PennyLaneAI/pennylane/pull/3874).
  [(#101)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/101)

* Updates to use the new call signature for `QuantumScript.get_opeartion`.
  [(#104)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/104)

### Contributors

Vincent Michaud-Rioux, Romain Moyard, Lee James O'Riordan

---

# Release 0.29.1

### Improvements

* Optimization updates to custatevector integration. E.g., creation of fewer cublas, cusparse and custatevec handles and fewer calls to small data transfers between host and device. [(#73)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/73)

### Contributors

Ania Brown (NVIDIA), Andreas Hehn (NVIDIA)

---

# Release 0.29.0

### Improvements

* Update `inv()` to `qml.adjoint()` in Python tests following recent changes in Pennylane.
 [(#88)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/88)

* Remove explicit Numpy requirement.
[(#90)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/90)

### Bug fixes

* Ensure early-failure rather than return of incorrect results from out of order probs wires.
[(#94)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/94)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Lee James O'Riordan, Shuli Shu

---

# Release 0.28.1

### Bug fixes

* Downgrade CUDA compiler for wheels to avoid compatibility issues with older runtimes.
 [(#87)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/87)

* Add header `unordered_map` to `util/cuda_helpers.hpp`.
 [(#86)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/86)

### Contributors

This release contains contributions from (in alphabetical order):

Lee James O'Riordan, Feng Wang

---

# Release 0.28.0

### New features since last release

* Add customized CUDA kernels for statevector initialization to cpp layer.
[(#70)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/70)

### Breaking changes

* Deprecate `_state` and `_pre_rotated_state` and refactor `syncH2D` and `syncD2H`.
[(#70)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/70)

The refactor on `syncH2D` and `syncD2H` allows users to explicitly access and update statevector data
on device when needed and could reduce the unnecessary memory allocation on host.

The workflow for `syncH2D` is:
```python
dev = qml.device('lightning.gpu', wires=3)
obs = qml.Identity(0) @ qml.PauliX(1) @ qml.PauliY(2)
obs1 = qml.Identity(1)
H = qml.Hamiltonian([1.0, 1.0], [obs1, obs])
state_vector = np.array([0.0 + 0.0j, 0.0 + 0.1j, 0.1 + 0.1j, 0.1 + 0.2j,
                0.2 + 0.2j, 0.3 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j,], dtype=np.complex64,)
dev.syncH2D(state_vector)
res = dev.expval(H)
```

The workflow for `syncD2H` is:
```python
dev = qml.device('lightning.gpu', wires=num_wires)
dev.apply([qml.PauliX(wires=[0])])
state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
dev.syncD2H(state_vector)
```

* Deprecate Python 3.7 wheels.
[(#75)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/75)

- Change the signature of the `DefaultQubit.signature` method from

  ```python
  def statistics(self, observables, shot_range=None, bin_size=None, circuit=None):
  ```

  to

  ```python
  def statistics(self, circuit: QuantumScript, shot_range=None, bin_size=None):
  ```

### Improvements

* `lightning.gpu` is decoupled from Numpy layer during initialization and execution
and change `lightning.gpu` to inherit from `QubitDevice` instead of `LightningQubit`.
[(#70)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/70)

* Add support for CI checks.
[(#76)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/76)

* Implement improved `stopping_condition` method, and make Linux wheel builds more performant.
[(#77)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/77)

### Bug fixes

* Fix wheel-builder to pin CUDA version to 11.8 instead of latest.
[(#83)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/83)

* Pin CMake to 3.24.x in wheel-builder to avoid Python not found error in CMake 3.25.
[(#75)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/75)

* Fix data copy method in the state() method.
[(#82)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/82)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Lee J. O'Riordan, Shuli Shu

---

# Release 0.27.0

### New features since last release

* Explicit support for `qml.SparseHamiltonian` using the adjoint gradient method.
  [(#72)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/72)

  This support allows users to explicitly make use of `qml.SparseHamiltonian` in expectation value calculations, and ensures the gradients can be taken efficiently.
  A user can now explicitly decide whether to decompose the Hamiltonian into separate Pauli-words, with evaluations happening over multiple GPUs, or convert the Hamiltonian directly to a sparse representation for evaluation on a single GPU. Depending on the Hamiltonian structure, a user may benefit from one method or the other.

  The workflow for decomposing a Hamiltonian is as:
  ```python
  obs_per_gpu = 1
  dev = qml.device("lightning.gpu", wires=num_wires, batch_obs=obs_per_gpu)

  H = sum([0.5*(i+1)*(qml.PauliZ(i)@qml.PauliZ(i+1)) for i in range(0, num_wires-1, 2)])

  @qml.qnode(dev, diff_method="adjoint")
  def circuit(params):
      for i in range(num_wires):
          qml.RX(params[i], i)
      return qml.expval(H)
  ```

  For the new `qml.SparseHamiltonian` support, the above script becomes:
  ```python
  dev = qml.device("lightning.gpu", wires=num_wires)
  H = sum([0.5*(i+1)*(qml.PauliZ(i)@qml.PauliZ(i+1)) for i in range(0, num_wires-1, 2)])
  H_sparse_matrix = qml.utils.sparse_hamiltonian(H, wires=range(num_wires))

  SpH = qml.SparseHamiltonian(H_sparse_matrix, wires=range(num_wires))

  @qml.qnode(dev, diff_method="adjoint")
  def circuit(params):
      for i in range(num_wires):
          qml.RX(params[i], i)
      return qml.expval(SpH)
  ```

* Enable building of python 3.11 wheels and upgrade python on CI/CD workflows to 3.8.
[(#71)](https://github.com/PennyLaneAI/pennylane-lightning/pull/71)

### Improvements

* Update `LightningGPU` device following changes in `LightningQubit` inheritance from `DefaultQubit` to `QubitDevice`.
[(#74)](https://github.com/PennyLaneAI/pennylane-lightning/pull/74)

### Bug fixes

* Ensure device fallback successfully carries through for 0 devices
[(#67)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/67)

* Fix void data type used in SparseSpMV
[(#69)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/69)

### Contributors

Amintor Dusko, Lee J. O'Riordan, Shuli Shu

---

# Release 0.26.2

### Bug fixes

* Fix reduction over batched & decomposed Hamiltonians in adjoint pipeline
[(#64)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/64)

### Contributors

Lee J. O'Riordan

---

# Release 0.26.1

### Bug fixes

* Ensure `qml.Hamiltonian` is auto-decomposed for the adjoint differentiation pipeline to avoid OOM errors.
[(#62)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/62)

### Contributors

Lee J. O'Riordan

---

# Release 0.26.0

### New features since last release

* Added native support for expval(H) in adjoint method. [(#52)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/52)

* Added cuSparse SpMV in expval(H) calculations. [(#52)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/52)

### Bug fixes

* Fix statistics method to support changes in qubit device API.
[(#55)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/55)

* Update Lightning tag to latest_release.
[(#51)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/51)

* Reintroduce dispatching support for `SingleExcitation` and `DoubleExcitation` gates in C++ layer.
[(#56)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/56)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Lee James O'Riordan, Shuli Shu

---

# Release 0.25.0

### New features since last release

* Added support for multi-GPU adjoint observable batching. [(#27)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/27)

  This new feature allows users to batch their gradients over observables using the `adjoint` method. Assuming multiple GPUs on a host-system, this functionality can be enabled by adding the `batch_obs=True` argument when creating a device, such as:

  ```python
  dev = qml.device("lightning.gpu", wires=6, batch_obs=True)
  ...
  @qml.qnode(dev, diff_method="adjoint")
  def circuit(params):
    for idx,w in enumerate(dev.wires):
      qml.RX(params[idx], w)
    return [qml.expval(qml.PauliZ(i))  for i in range(dev.num_wires)]
  ```
For comparison, we can re-examine the benchmark script from the [Lightning GPU PennyLane blog post](https://pennylane.ai/blog/2022/07/lightning-fast-simulations-with-pennylane-and-the-nvidia-cuquantum-sdk/). Comparing with and without the multi-GPU supports on a machine with 4 A100 40GB GPUs shows a significant improvement over the single GPU run-times.

![image](https://user-images.githubusercontent.com/858615/184025758-7adeb433-5f7b-451a-bc72-ee3f7e321c49.png)

### Bug fixes

* Fix `test-cpp` Makefile rule to run the correct GPU-compiled executable [(#42)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/42)

* Updates to ensure compatibility with cuQuantum 22.0.7. [(#38)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/38)

* Bugfix for IsingZZ generator indices and adjoint tests. [(#40)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/40)

* Add RAII wrapping of custatevec handles to avoid GPU memory leaking of CUDA contexts [(#41)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/41)

* Updated capabilities dictionary to ensure finite-shots support is set to `True`. [(#34)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/34)

### Contributors

Lee James O'Riordan, Trevor Vincent

---

# Release 0.24.1

### Bug fixes

* Ensure diagonalizing gates are applied before sampling on GPU. [(#36)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/36)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Lee James O'Riordan

---

# Release 0.24.0

### New features since last release

* Add a cuquantum-backed finite shot sampling method. [(#21)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/21)

* Add more cuquantum-backed gates (IsingXX/YY/ZZ, MultiRZ, SingleExcitation, ...). [(#28)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/28)

### Breaking changes

* Update `lightning.gpu` to build and run with the recent changes in `lightning.qubit`. This updates the GCC version and moves to C++20. [(#30)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/30)

### Improvements

* LightningGPU can be installed in-place to a Python environment via `pip install -e`. [(#26)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/26)

* CPU-only warnings are now more visible. [(#23)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/23)

### Bug fixes

* Fix jacobian tape with state preparation. [(#32)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/32)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Amintor Dusko, Chae-Yeun Park, Lee James O'Riordan, and Trevor Vincent

---

# Release 0.23.0

### Improvements

* Update builder and cuQuantum SDK support [(#10)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/10).

### Contributors

This release contains contributions from (in alphabetical order):
Ali Asadi, and Lee James O'Riordan

---

# Release 0.22.1

### Improvements

* Add `Identity` support [(#8)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/8).

### Contributor

This release contains contributions from (in alphabetical order):
Lee James O'Riordan

---

# Release 0.22.0

* Formal release with NVIDIA cuQuantum SDK 1.0 support.

### Improvements

* Release semantic versioning matches PennyLane current release versioning.
[(#6)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/6)

### Bug fixes

* This release updates the cuQuantum function calls to match the SDK 1.0 release.
[(#6)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/6)

### Contributors

This release contains contributions from (in alphabetical order):
Ali Asadi, and Lee James O'Riordan

---

# Release 0.1.0

 * Initial release. The PennyLane-Lightning-GPU device adds support for CUDA-capable GPU simulation through use of the NVIDIA cuQuantum SDK.
This release supports all base operations, including the adjoint differentation method for expectation value calculations.

This device can be installed using `pip install pennylane-lightning[gpu]`, and requires both a NVIDIA CUDA installation and the cuQuantum SDK to operate. If the host system does not provide sufficient support, the device will fall-back to CPU-only operation.


As an example, the new device may be used as follows to calculate the forward pass of a circuit with 2 strongly-entangling layers and an expectation value per wire:

```python
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers
from numpy.random import random

n_wires = 20
n_layers = 2

dev = qml.device("lightning.gpu", wires=n_wires)

@qml.qnode(dev, diff_method=None)
def circuit(weights):
    StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

params = np.random.random(StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
result = circuit(params)
```

One can also obtain the Jacobian of the above circuit using the adjoint differentiation method as:

```python
@qml.qnode(dev, diff_method="adjoint")
def circuit_adj(weights):
    StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

jac = qml.jacobian(circuit_adj)(params)
```

For the Jacobian evaluation, we see significant speed-ups for a single NVIDIA A100 GPU compared to the multi-threaded CPU-only implementation

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane-lightning-gpu/main/doc/_static/lightning_gpu_initial_bm.png" width=50%/>

This release contains contributions from (in alphabetical order):

Ali Asadi, Lee James O'Riordan, Trevor Vincent
