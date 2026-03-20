# -*- coding: utf-8 -*-
"""Package build configuration for Fast Poisson Image Editing."""

import importlib.machinery
import os
import shutil
import subprocess
import sys
from multiprocessing import cpu_count
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def get_version() -> str:
    """Read the package version from ``fpie.__init__``."""
    # https://packaging.python.org/guides/single-sourcing-package-version/
    with open(os.path.join("fpie", "__init__.py"), "r") as f:
        init = f.read().split()
    return init[init.index("__version__") + 2][1:-1]


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    """Describe a CMake-built extension module."""

    def __init__(self, name, sourcedir=""):
        """Record the CMake source directory for the extension."""
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Provide setuptools hooks for building CMake extensions."""

    def initialize_options(self) -> None:
        """Initialize setuptools and track CMake output directories."""
        super().initialize_options()
        self._cmake_output_dirs = {}

    def _iter_native_artifacts(self, *search_roots: Path):
        seen = set()
        for output_dir in search_roots:
            if not output_dir.exists():
                continue
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                for artifact in sorted(output_dir.rglob(f"core_*{suffix}")):
                    if artifact.is_file() and artifact not in seen:
                        seen.add(artifact)
                        yield artifact

    def build_extension(self, ext):
        """Build an extension with CMake and ``make``."""
        extdir = Path(self.build_lib) / ext.name
        self._cmake_output_dirs[ext.name] = extdir

        # required for auto-detection of auxiliary "native" libs
        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.path.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build_args = ["-j", f"{cpu_count()}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        extdir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.check_call(
                ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
            )
            subprocess.check_call(["make"] + build_args, cwd=self.build_temp)
            # Some CMake generators place per-target outputs under nested build
            # directories, so copy any produced backend modules into build_lib.
            for artifact in self._iter_native_artifacts(Path(self.build_temp)):
                shutil.copy2(artifact, extdir / artifact.name)
        except Exception as e:
            print(f"{type(e)}: {e}")
            pass

    def copy_extensions_to_source(self) -> None:
        """Copy built native artifacts into the source package."""
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            if not isinstance(ext, CMakeExtension):
                self.copy_extension_to_source(ext)
                continue

            output_dir = self._cmake_output_dirs.get(ext.name)
            if output_dir is None:
                continue

            package_dir = Path(build_py.get_package_dir(ext.name))
            package_dir.mkdir(parents=True, exist_ok=True)
            for artifact in self._iter_native_artifacts(output_dir):
                shutil.copy2(artifact, package_dir / artifact.name)


def get_description():
    """Read the project README for the long description."""
    with open("README.md", encoding="utf8") as f:
        return f.read()


setup(
    name="fpie",
    version=get_version(),
    author="Jiayi Weng",
    author_email="trinkle23897@gmail.com",
    url="https://github.com/Trinkle23897/Fast-Poisson-Image-Editing",
    license="MIT",
    description="A fast poisson image editing algorithm implementation.",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.14",
    packages=find_packages(exclude=["tests", "tests.*"]),
    entry_points={
        "console_scripts": ["fpie=fpie.cli:main", "fpie-gui=fpie.gui:main"],
    },
    install_requires=[
        "cmake>=3.5",
        "opencv-python-headless>=4.2",
        "numpy>=1.18",
        # these packages are universal
        "taichi>=1.0",
        "numba>=0.51",
    ],
    extras_require={
        "dev": [
            "ruff",
            "mypy",
        ],
        "mpi": ["mpi4py>=3.1"],  # cannot install on mac M1
    },
    ext_modules=[CMakeExtension("fpie")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
