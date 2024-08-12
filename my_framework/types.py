from typing import Any, Optional, Union
from numpy import typing as npt
import numpy as np

NDArray = Union[npt.NDArray]

try:
    import cupy as cp
    NDArray = Union[npt.NDArray, cp.ndarray]
except ImportError:
    NDArray = Union[npt.NDArray]

