PennyLane-Lightning-GPU Plugin
##############################

.. image:: https://readthedocs.com/projects/xanaduai-pennylane-lightning-gpu/badge/?version=latest&style=flat-square
    :alt: Read the Docs
    :target: https://docs.pennylane.ai/projects/lightning-gpu

.. image:: https://img.shields.io/pypi/v/PennyLane-Lightning-GPU.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-Lightning-GPU

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-Lightning-GPU.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-Lightning-GPU

.. header-start-inclusion-marker-do-not-remove

The `PennyLane-Lightning-GPU <https://github.com/PennyLaneAI/pennylane-lightning-gpu>`_ plugin extends the `Pennylane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ state-vector simulator written in C++, and offloads to the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ for GPU accelerated circuit simulation.

`PennyLane <https://docs.pennylane.ai>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

.. header-end-inclusion-marker-do-not-remove


Features
========

* Combine the NVIDIA cuQuantum SDK high-performance GPU simulator library with PennyLane's
  automatic differentiation and optimization.

* Direct support for GPU-enabled quantum gradients with the `adjoint differentiation method <https://docs.pennylane.ai/en/stable/introduction/interfaces.html#simulation-based-differentiation>`_.

.. installation-start-inclusion-marker-do-not-remove


Installation
============

PennyLane-Lightning-GPU requires Python version 3.8 and above. It can be installed using ``pip``:

.. code-block:: console

    pip install pennylane-lightning[gpu]

Use of PennyLane-Lightning-GPU also requires explicit installation of the NVIDIA cuQuantum SDK. The SDK library directory may be provided on the ``LD_LIBRARY_PATH`` environment variable, or the SDK Python package may be installed within the Python environment ``site-packages`` directory using ``pip`` or ``conda``. Please see the `cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ install guide for more information.

To build a wheel from the package sources using the direct SDK path:

.. code-block:: console

    cmake -BBuild -DENABLE_CLANG_TIDY=on -DCUQUANTUM_SDK=<path to sdk>
    cmake --build ./Build --verbose
    python -m pip install wheel
    python setup.py build_ext --cuquantum=<path to sdk>
    python setup.py bdist_wheel


To build using the PyPI/Conda installed cuQuantum package:

.. code-block:: console

    python -m pip install wheel cuquantum
    python setup.py build_ext
    python setup.py bdist_wheel

The built wheel can now be installed as:

.. code-block:: console

    python -m pip install ./dist/PennyLane_Lightning_GPU-*.whl

To simplify the build, we recommend using the following containerized build process, which creates `manylinux2014 <https://github.com/pypa/manylinux>`_ compatible wheels.


Build locally with Docker
-------------------------

To build using Docker, run the following from the project root directory:

.. code-block:: console

    docker build . -f ./docker/Dockerfile -t "lightning-gpu-wheels"

This will build a Python wheel for Python 3.8 up to 3.11 inclusive, and be manylinux2014 (glibc 2.17) compatible.
To acquire the built wheels, use:

.. code-block:: console

    docker run -v `pwd`:/io -it lightning-gpu-wheels cp -r ./wheelhouse /io

which mounts the current working directory, and copies the wheelhouse directory from the image to the local directory.
For licensing information, please view ``docker/README.md``.

Build PennyLane-Lightning-GPU with multi-node/multi-gpu support
---------------------------------------------------------------

Use of PennyLane-Lightning-GPU with multi-node/multi-gpu support also requires explicit installation of the ``NVIDIA cuQuantum SDK`` (current supported 
cuQuantum version: `cuquantum-cu11 <https://pypi.org/project/cuquantum-cu11/>`_), ``mpi4py`` and ``CUDA-aware MPI`` (Message Passing Interface). 
``CUDA-aware MPI`` allows data exchange between GPU memory spaces of different nodes without the need for CPU-mediated transfers. Both ``MPICH`` 
and ``OpenMPI`` libraries are supported, provided they are compiled with CUDA support. Path to the ``libmpi.so`` should be added to the ``LD_LIBRARY_PATH`` environment variable.
It's recommended to install ``NVIDIA cuQuantum SDK`` and ``mpi4py`` Python package within the Python environment ``site-packages`` directory using ``pip`` or ``conda``. 
Please see the `cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ , `mpi4py <https://mpi4py.readthedocs.io/en/stable/install.html>`_, 
`MPICH <https://www.mpich.org/static/downloads/4.1.1/mpich-4.1.1-README.txt>`_, or `OpenMPI <https://www.open-mpi.org/faq/?category=buildcuda>`_ install guide for more information.

To build a wheel with multi-node/multi-gpu support from the package sources using the direct SDK path:

.. code-block:: console

    cmake -BBuild -DENABLE_CLANG_TIDY=on -DPLLGPU_ENABLE_MPI=on -DCUQUANTUM_SDK=<path to sdk>
    cmake --build ./Build --verbose
    python -m pip install wheel
    python setup.py build_ext --define="PLLGPU_ENABLE_MPI=ON" --cuquantum=<path to sdk>
    python setup.py bdist_wheel


The built wheel can now be installed as:

.. code-block:: console

    python -m pip install ./dist/PennyLane_Lightning_GPU-*.whl

Testing
=======

Test PennyLane-Lightning-GPU
-----------------------------------------------------------------

To test that the plugin is working correctly you can test the Python code within the cloned
repository:

.. code-block:: console

    make test-python

while the C++ code can be tested with

.. code-block:: console

    make test-cpp


Please refer to the `GPU plugin documentation <https://docs.pennylane.ai/projects/lightning-gpu>`_ as
well as to the `CPU documentation <https://docs.pennylane.ai/projects/lightning>`_ and 
`PennyLane documentation <https://pennylane.readthedocs.io/>`_ for further references.

Test PennyLane-Lightning-GPU with multi-node/multi-gpu support
---------------------------------------------------------------

To test that the plugin is working correctly you can test the Python code within the cloned
repository:

.. code-block:: console

    mpirun -np 2 python -m pytest mpitests --tb=short

while the C++ code can be tested with

.. code-block:: console

    rm -rf ./BuildTests
    cmake . -BBuildTests -DBUILD_TESTS=1 -DPLLGPU_BUILD_TESTS=1 -DPLLGPU_ENABLE_MPI=On -DCUQUANTUM_SDK=<path to sdk>
    cmake --build ./BuildTests --verbose
    mpirun -np 2 ./BuildTests/pennylane_lightning_gpu/src/tests/mpi_runner

.. installation-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributors to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane-lightning-gpu
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-lightning-gpu/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove


License
=======

The PennyLane-Lightning-GPU plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_. 
The PennyLane-Lightning-GPU plugin makes use of the NVIDIA cuQuantum SDK headers to 
enable the device bindings to PennyLane, which are held to their own respective license.

.. license-end-inclusion-marker-do-not-remove
