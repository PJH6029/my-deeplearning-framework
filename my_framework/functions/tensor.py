from typing import Optional, Union
import numpy as np

import my_framework.core.base as base
from my_framework.types import NDArray
from my_framework import utils
import my_framework.cuda as cuda

class Reshape(base.Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
    
    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        return reshape(gy, self.x_shape)

def reshape(x: Union[base.Variable, NDArray], shape: tuple[int, ...]) -> base.Variable:
    if x.shape == shape:
        return base.as_variable(x)
    return Reshape(shape)(x)

class Transpose(base.Function):
    def __init__(self, axes: Optional[tuple[int, ...]] = None) -> None:
        self.axes = axes
    
    def forward(self, x: NDArray) -> NDArray:
        y = x.transpose(self.axes)
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        if self.axes is None:
            return transpose(gy)
        
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, axes=inv_axes)
    
def transpose(x: Union[base.Variable, NDArray], axes: Optional[tuple[int, ...]] = None) -> base.Variable:
    return Transpose(axes)(x)

# =============================================================================
# Advanced Functions
# =============================================================================
class Sum(base.Function):
    def __init__(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x: Union[base.Variable, NDArray], axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> base.Variable:
    return Sum(axis, keepdims)(x)


class SumTo(base.Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        
    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x: Union[base.Variable, NDArray], shape: tuple[int, ...]) -> base.Variable:
    if x.shape == shape:
        return base.as_variable(x)
    return SumTo(shape)(x)

class BroadcastTo(base.Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
    
    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x: base.Variable, shape: tuple[int, ...]) -> base.Variable:
    if x.shape == shape:
        return base.as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(base.Function):
    def forward(self, x: NDArray, W: NDArray) -> NDArray:
        y = x.dot(W) # handle both numpy and cupy
        return y
    
    def backward(self, gy: base.Variable) -> tuple[base.Variable, base.Variable]:
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x: Union[base.Variable, NDArray], W: Union[base.Variable, NDArray]) -> base.Variable:
    return MatMul()(x, W)


def linear_simple(x: base.Variable, W: base.Variable, b: Optional[base.Variable] = None) -> base.Variable:
    t = matmul(x, W)
    if b is None:
        return t
    
    y = t + b
    t.data = None # aggressive buffer release
    return y

class Linear(base.Function):
    def forward(self, x: NDArray, W: NDArray, b: Optional[NDArray] = None) -> NDArray:
        y = x.dot(W)
        if b is not None:
            y += b
        return y
    
    def backward(self, gy: base.Variable) -> tuple[base.Variable, base.Variable, Optional[base.Variable]]:
        x, W, b = self.inputs
        gb = None if (b is None or b.data is None) else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x: Union[base.Variable, NDArray], W: Union[base.Variable, NDArray], b: Optional[Union[base.Variable, NDArray]] = None) -> base.Variable:
    return Linear()(x, W, b)

def dropout(x: Union[base.Variable, NDArray], dropout_ratio: float = 0.5) -> base.Variable:
    x = base.as_variable(x)
    if base.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 / (1.0 - dropout_ratio), dtype=x.dtype)
        return x * mask * scale
    else:
        return x

def expand_dims(x: Union[base.Variable, NDArray], axis: int) -> base.Variable:
    x = base.as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))

def flatten(x: Union[base.Variable, NDArray]) -> base.Variable:
    return reshape(x, (x.shape[0], -1))
