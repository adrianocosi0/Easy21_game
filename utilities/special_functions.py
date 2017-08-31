import numpy as np

def inv0(c):
    """ ignore / 0, inv0([-1, 0, 3]) -> [-1, 0, 1/3] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = c**-1
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
