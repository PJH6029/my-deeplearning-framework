from typing import Optional, Union
import numpy as np

import my_framework.core.base as base
from my_framework.types import NDArray
from my_framework import utils
import my_framework.functions as F
from my_framework import cuda

def sigmoid_simple(x: Union[base.Variable, NDArray]) -> NDArray:
    x_var = base.as_variable(x)
    return 1 / (1 + F.exp(-x_var))


class Sigmoid(base.Function):
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5 # Better implementation
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x: Union[base.Variable, NDArray]) -> base.Variable:
    return Sigmoid()(x)


class ReLU(base.Function):
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        return xp.maximum(x, 0.0)

    def backward(self, gy: base.Variable) -> base.Variable:
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x: Union[base.Variable, NDArray]) -> base.Variable:
    return ReLU()(x)


def softmax_simple(x: Union[base.Variable, NDArray], axis: int = 1) -> base.Variable:
    # redundant computation graph.
    x = base.as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y, axis=axis, keepdims=True)
    return y / sum_y

class Softmax(base.Function):
    def __init__(self, axis: int = 1):
        self.axis = axis
        
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        y = xp.exp(x - x.max(axis=self.axis, keepdims=True))
        y /= y.sum(axis=self.axis, keepdims=True)
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x: Union[base.Variable, NDArray], axis: int = 1) -> base.Variable:
    return Softmax(axis)(x)

class LogSoftmax(base.Function):
    def __init__(self, axis: int = 1):
        self.axis = axis
        
    def forward(self, x: NDArray) -> NDArray:
        log_z = utils.logsumexp(x, axis=self.axis)
        y = x - log_z
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        y = self.outputs[0]()
        gx = gy - F.exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx

def log_softmax(x: Union[base.Variable, NDArray], axis: int = 1) -> base.Variable:
    return LogSoftmax(axis)(x)


class LeakyReLU(base.Function):
    def __init__(self, slope: float = 0.2):
        self.slope = slope
    
    def forward(self, x: NDArray) -> NDArray:
        y = x.copy()
        y[y < 0] *= self.slope
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        x, = self.inputs
        mask = (x.data >= 0).astype(gy.dtype)
        mask[mask < 0] = self.slope
        gx = gy * mask
        return gx

def leaky_relu(x: Union[base.Variable, NDArray], slope: float = 0.2) -> base.Variable:
    return LeakyReLU(slope)(x)
