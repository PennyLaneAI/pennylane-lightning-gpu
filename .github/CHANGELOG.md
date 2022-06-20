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
