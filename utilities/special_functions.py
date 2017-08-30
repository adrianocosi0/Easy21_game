import numpy as np

def inv0(c):
    """ ignore / 0, inv0([-1, 0, 3]) -> [-1, 0, 1/3] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = c**-1
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c
