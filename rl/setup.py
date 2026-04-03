"""Build Cython extensions + optional CUDA extension for G(n,m) RL."""

import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# --- Cython extensions (always built) ---
cython_extensions = [
    Extension(
        "utils._fast_features",
        ["utils/_fast_features.pyx"],
        include_dirs=[np.get_include()],
    ),
]

cython_modules = cythonize(
    cython_extensions,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    },
)

# --- CUDA extension (optional, requires torch + CUDA) ---
cuda_modules = []
try:
    import torch
    if torch.cuda.is_available():
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        cuda_modules = [
            CUDAExtension(
                "utils.gnm_cuda",
                [
                    "utils/csrc/bindings.cu",
                    "utils/csrc/kernels.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "-gencode=arch=compute_80,code=sm_80",   # Ampere
                        "-gencode=arch=compute_89,code=sm_89",   # Ada
                    ],
                },
            ),
        ]

        # Check for Blackwell support (sm_120) in nvcc
        # Only add if CUDA toolkit supports it
        try:
            cuda_version = torch.version.cuda
            if cuda_version and tuple(int(x) for x in cuda_version.split('.')[:2]) >= (12, 8):
                cuda_modules[0].extra_compile_args["nvcc"].append(
                    "-gencode=arch=compute_120,code=sm_120"
                )
        except Exception:
            pass

except ImportError:
    pass
except Exception:
    pass

# --- Combined setup ---
all_modules = cython_modules + cuda_modules

cmdclass = {}
if cuda_modules:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass["build_ext"] = BuildExtension

setup(
    ext_modules=all_modules,
    cmdclass=cmdclass,
)
