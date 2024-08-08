from typing import Optional, Union
import numpy as np

import my_framework.core.base as base
from my_framework.types import NDArray
from my_framework import utils
import my_framework.functions as F

def sigmoid_simple(x: Union[base.Variable, NDArray]) -> NDArray:
    x_var = base.as_variable(x)
    return 1 / (1 + F.exp(-x_var))


class Sigmoid(base.Function):
    def forward(self, x: NDArray) -> NDArray:
        # y = 1 / (1 + np.exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5 # Better implementation
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x: Union[base.Variable, NDArray]) -> base.Variable:
    return Sigmoid()(x)
