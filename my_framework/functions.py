import numpy as np
from my_framework.types import *
from my_framework.core import Variable, Function, as_variable
import my_framework
from my_framework import utils

# =============================================================================
# Basic Functions
# =============================================================================
class Sin(Function):
    def forward(self, x: NDArray) -> NDArray:
        y = np.sin(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * cos(x)
        return gx
    
def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x: NDArray) -> NDArray:
        y = np.cos(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: NDArray) -> NDArray:
        y = np.tanh(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx
    
def tanh(x):
    return Tanh()(x)


# =============================================================================
# Tensor Functions
# =============================================================================
class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
    
    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)

def reshape(x: Union[Variable, NDArray], shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self, axes: Optional[tuple[int, ...]] = None) -> None:
        self.axes = axes
    
    def forward(self, x: NDArray) -> NDArray:
        y = np.transpose(x, axes=self.axes)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)
        
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, axes=inv_axes)
    
def transpose(x: Union[Variable, NDArray], axes: Optional[tuple[int, ...]] = None) -> Variable:
    return Transpose(axes)(x)

# =============================================================================
# Advanced Functions
# =============================================================================
class Sum(Function):
    def __init__(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x: Union[Variable, NDArray], axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> Variable:
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        
    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x: Union[Variable, NDArray], shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class BroadcastTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
    
    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, x: NDArray, W: NDArray) -> NDArray:
        y = np.dot(x, W)
        return y
    
    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x: Union[Variable, NDArray], W: Union[Variable, NDArray]) -> Variable:
    return MatMul()(x, W)

# =============================================================================
# Loss Functions
# =============================================================================
def mean_squared_error_naive(x0, x1):
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)

class MeanSquaredError(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        diff = x0 - x1
        y = (diff ** 2).mean()
        return y
        
    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0: Union[Variable, NDArray], x1: Union[Variable, NDArray]) -> Variable:
    return MeanSquaredError()(x0, x1)