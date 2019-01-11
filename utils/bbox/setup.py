import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
numpy_include = np.get_include()
setup(ext_modules=cythonize("bbox.pyx"),include_dirs=[numpy_include])
setup(ext_modules=cythonize("nms.pyx"),include_dirs=[numpy_include])