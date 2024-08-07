from typing import Any, Optional, Union
from numpy import typing as npt
import my_framework

try:
    # from cupy.t TODO
    pass
except ImportError:
    pass


# VariableOutput = Union[core.Variable, ]

NDArray = Union[npt.NDArray]

# InputType = Union[NDArray, Variable] TODO

# FunctionParam = Union[NDArray, my_framework.core.Variable]