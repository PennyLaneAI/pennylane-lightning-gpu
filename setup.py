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

import os
import platform
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages

if not os.getenv("READTHEDOCS"):

    from setuptools import Extension
    from setuptools.command.build_ext import build_ext

    class CMakeExtension(Extension):
        def __init__(self, name, sourcedir=""):
            Extension.__init__(self, name, sources=[])
            self.sourcedir = Path(sourcedir).absolute()

    class CMakeBuild(build_ext):
        """
        This class is based upon the build infrastructure of Pennylane-Lightning.
        """

        user_options = build_ext.user_options + [
            ("define=", "D", "Define variables for CMake"),
            ("cuquantum=", None, "Path to cuQuantum SDK"),
            ("verbosity", "V", "Increase CMake build verbosity"),
        ]

        def initialize_options(self):
            super().initialize_options()
            self.define = None
            self.cuquantum = os.getenv("CUQUANTUM_SDK", None)
            self.verbosity = ""

        def finalize_options(self):
            # Parse the custom CMake options and store them in a new attribute
            defines = [] if self.define is None else self.define.split(";")
            self.cmake_defines = [f"-D{define}" for define in defines]
            if self.cuquantum is not None:
                self.cmake_defines.append(f"-DCUQUANTUM_SDK={self.cuquantum}")
            if self.verbosity != "":
                self.verbosity = "--verbose"

            super().finalize_options()

        def build_extension(self, ext: CMakeExtension):
            extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.absolute())
            debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
            cfg = "Debug" if debug else "RelWithDebInfo"
            ninja_path = str(shutil.which("ninja"))

            # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
            configure_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
                "-GNinja",
                f"-DCMAKE_MAKE_PROGRAM={ninja_path}",
                f"-DENABLE_OPENMP=OFF",
                *(self.cmake_defines),
            ]

            build_args = []

            if not platform.system() == "Linux":
                raise RuntimeError(f"Unsupported '{platform.system()}' platform")

            if not Path(self.build_temp).exists():
                os.makedirs(self.build_temp)

            subprocess.check_call(
                ["cmake", str(ext.sourcedir)] + configure_args, cwd=self.build_temp
            )
            subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


with open("pennylane_lightning_gpu/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "ninja",
    "wheel",
    "cmake",
    "numpy",
    "pennylane-lightning>=0.22",
    "pennylane>=0.22",
]

info = {
    "name": "PennyLane-Lightning-GPU",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/pennylane-lightning-gpu",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "package_data": {"pennylane_lightning_gpu": ["src/*"]},
    "entry_points": {
        "pennylane.plugins": [
            "lightning.gpu = pennylane_lightning_gpu:LightningGPU",
        ],
    },
    "description": "PennyLane-Lightning-GPU plugin",
    "long_description": open("README.rst").read(),
    "long_description_content_type": "text/x-rst",
    "provides": ["pennylane_lightning_gpu"],
    "install_requires": requirements,
    "ext_package": "pennylane_lightning_gpu",
}

if not os.getenv("READTHEDOCS"):
    info["ext_modules"] = [CMakeExtension("lightning_gpu_qubit_ops")]
    info["cmdclass"] = {"build_ext": CMakeBuild}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
