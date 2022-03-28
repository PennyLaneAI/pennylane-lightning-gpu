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
