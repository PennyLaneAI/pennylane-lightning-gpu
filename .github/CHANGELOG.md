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

### Breaking changes

### Improvements

### Documentation

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

### Bug fixed

* Fix `test-cpp` Makefile rule to run the correct GPU-compiled executable [(#42)](https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/42)

### Bug fixes

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

### Documentation

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
