from typing import Union
import numpy as np
from my_framework.types import *

gpu_enable = True
XPNDArray = Union[NDArray]

try:
    import cupy as cp
    cupy = cp
    XPNDArray = Union[NDArray, cp.ndarray]
except ImportError:
    gpu_enable = False
    XPNDArray = Union[NDArray]

from my_framework.core import Variable


def get_array_module(x: Union[Variable, XPNDArray]) -> Any:
    if isinstance(x, Variable):
        x = x.data
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp

def as_numpy(x: Union[Variable, XPNDArray]) -> NDArray:
    if isinstance(x, Variable):
        x = x.data
    
    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)

def as_cupy(x: Union[Variable, XPNDArray]) -> XPNDArray:
    if isinstance(x, Variable):
        x = x.data
    if not gpu_enable:
        raise Exception("Cannot convert to cupy")
    return cp.asarray(x)