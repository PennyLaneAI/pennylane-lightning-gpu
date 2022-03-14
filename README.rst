PennyLane-Lightning-GPU Plugin
##############################

.. image:: https://img.shields.io/readthedocs/pennylane-lightning-gpu.svg?logo=read-the-docs&style=flat-square
    :alt: Read the Docs
    :target: https://pennylane-lightning-gpu.readthedocs.io

.. image:: https://img.shields.io/pypi/v/PennyLane-Lightning-GPU.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-Lightning-GPU

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-Lightning-GPU.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-Lightning-GPU

.. header-start-inclusion-marker-do-not-remove

The `PennyLane-Lightning-GPU <https://github.com/PennyLaneAI/pennylane-lightning-gpu>`_ plugin extends the `Pennylane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ state-vector simulator written in C++, and offloads to the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ for GPU accelerated circuit simulation.

`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

.. header-end-inclusion-marker-do-not-remove


Features
========

* Combine the NVIDIA cuQuantum SDK high-performance GPU simulator library with PennyLane's
  automatic differentiation and optimization.

* Direct support for GPU-enabled quantum gradients with the `adjoint differentiation method <https://pennylane.readthedocs.io/en/stable/introduction/interfaces.html#simulation-based-differentiation>`_.

.. installation-start-inclusion-marker-do-not-remove


Installation
============

PennyLane-Lightning-GPU requires Python version 3.7 and above. It can be installed using ``pip``:

.. code-block:: console

    pip install pennylane-lightning[gpu]

To build the C++ module from source:

.. code-block:: console

    cmake -BBuild -DENABLE_CLANG_TIDY=on -DCUQUANTUM_SDK=<path to sdk>
    cmake --build ./Build --verbose


An Python wheel can be built using:

.. code-block:: console

    python -m pip install wheel
    python setup.py build_ext --cuquantum=<path to sdk>
    python setup.py bdist_wheel


To simplify the build process, we recommend using the following containerized build process.


Build locally with Docker
-------------------------

To build using Docker, run the following from the project root directory:

.. code-block:: console

    docker build . -f ./docker/Dockerfile -t "lightning-gpu-wheels"

This will build a Python wheel for Python 3.7 up to 3.10 inclusive, and be manylinux2014 (glibc 2.17) compatible.
To acquire the built wheels, use:

.. code-block:: console

    docker run -v `pwd`:/io -it lightning-gpu-wheels cp -r ./wheelhouse /io

which mounts the current working directory, and copies the wheelhouse directory from the image to the local directory.
For licensing information, please view ``docker/README.md``.


Testing
-------

To test that the plugin is working correctly you can test the Python code within the cloned
repository:

.. code-block:: console

    make test-python

while the C++ code can be tested with

.. code-block:: console

    make test-cpp


Please refer to the `GPU plugin documentation <https://pennylane-lightning-gpu.readthedocs.io/>`_ as
well as to the `CPU documentation <https://pennylane-lightning.readthedocs.io/>`_ and 
`PennyLane documentation <https://pennylane.readthedocs.io/>`_ for further references.

.. installation-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

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
