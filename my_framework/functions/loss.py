from typing import Optional, Union
import numpy as np

import my_framework.core.base as base
from my_framework.types import NDArray

def mean_squared_error_naive(x0, x1):
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)

class MeanSquaredError(base.Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        diff = x0 - x1
        y = (diff ** 2).mean()
        return y
        
    def backward(self, gy: base.Variable) -> tuple[base.Variable, base.Variable]:
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0: Union[base.Variable, NDArray], x1: Union[base.Variable, NDArray]) -> base.Variable:
    return MeanSquaredError()(x0, x1)