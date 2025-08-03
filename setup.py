from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="lens_solver_cy",
        sources=["lens_solver_cy.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="sl_inference",
    ext_modules=cythonize(extensions, language_level=3),
)
