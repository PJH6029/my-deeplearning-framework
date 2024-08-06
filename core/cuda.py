import numpy as np
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False

from typing import Any, Union

from core.core_simple import Variable
from core.types import *

def get_array_module(x: Union[Variable, NDArray]) -> Any:
    if isinstance(x, Variable):
        x = x.data
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp