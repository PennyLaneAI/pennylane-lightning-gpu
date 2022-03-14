Lightning-GPU device
======================

The ``lightning.gpu`` device is an extension of PennyLane's built-in ``lightning.qubit`` device.
It extends the CPU-focused Lightning simulator to run using the NVIDIA cuQuantum SDK, enabling GPU-accelerated simulation of quantum state-vector evolution.

A ``lightning.gpu`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.gpu", wires=2)

If the NVIDIA cuQuantum libraries are available, the above device will allow all operations to be perfomed on a CUDA capable GPU of generation SM 7.0 (Volta) and greater. If the libraries are not correctly installed, or available on path, the device will fall-back to ``lightning.qubit`` and perform all simulation on the CPU.

The ``lightning.gpu`` device also directly supports quantum circuit gradients using the adjoint differentiation method. This can be enabled at the PennyLane QNode level with:

.. code-block:: python

    qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        ...


Supported operations and observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.CNOT
    ~pennylane.CRot
    ~pennylane.CRX
    ~pennylane.CRY
    ~pennylane.CRZ
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.PhaseShift
    ~pennylane.ControlledPhaseShift
    ~pennylane.QubitStateVector
    ~pennylane.Rot
    ~pennylane.RX
    ~pennylane.RY
    ~pennylane.RZ
    ~pennylane.S
    ~pennylane.T

.. raw:: html

    </div>

**Supported observables:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.Hadamard
    ~pennylane.Identity
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ

.. raw:: html

    </div>
