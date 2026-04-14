"""编译 Cython 加速模块: python setup_cython.py build_ext --inplace"""
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        "_solver_fast.pyx",
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
)
