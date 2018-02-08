cimport cython
import numpy as np
cimport numpy as np


NP_FLOAT = np.float
NP_INT = np.int
ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t

def cy_argmax(np.ndarray[FLOAT_t, ndim=2] a):
    """Get the argmax of a on axis 0, **KNOWING** it is NON-negative"""
    cdef unsigned int rows = a.shape[0]
    cdef unsigned int cols = a.shape[1]
    cdef np.ndarray[INT_t  , ndim=1] idx  = np.zeros(cols, dtype=NP_INT)
    cdef np.ndarray[FLOAT_t, ndim=1] vals = np.zeros(cols, dtype=NP_FLOAT)

    for row in range(rows):
        for col in range(cols):
            if a[row,col] > vals[col]:
                vals[col] = a[row,col]
                idx[col] = row
    return idx
