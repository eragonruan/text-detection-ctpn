import numpy as np
import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.core import setup
from Cython.Build import cythonize
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


from distutils.core import setup
from Cython.Build import cythonize


ext_modules = [
    Extension(
        "utils.bbox",
        ["bbox.pyx"]
	),
    Extension(
        "utils.nms",
        ["nms.pyx"]
    ),]

setup(ext_modules=cythonize(ext_modules))

