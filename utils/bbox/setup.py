from distutils.core import setup

import numpy as np
from Cython.Build import cythonize

numpy_include = np.get_include()
setup(ext_modules=cythonize("cython_bbox.pyx"), include_dirs=[numpy_include])
setup(ext_modules=cythonize("cython_nms.pyx"), include_dirs=[numpy_include])
