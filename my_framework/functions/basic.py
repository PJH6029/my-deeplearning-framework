from typing import Union
import numpy as np

from my_framework.core import Variable, Function, as_variable, as_array
from my_framework.types import NDArray
from my_framework import utils
import my_framework.functions as F
import my_framework.cuda as cuda

# =============================================================================
# trigonometric
# =============================================================================
class Sin(Function):
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * cos(x)
        return gx
    
def sin(x: Union[Variable, NDArray]) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x: Union[Variable, NDArray]) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx
    
def tanh(x: Union[Variable, NDArray]) -> Variable:
    return Tanh()(x)


class Exp(Function):
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        return gy * y

def exp(x: Union[Variable, NDArray]) -> Variable:
    return Exp()(x)


class Log(Function):
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return gy / x
    
def log(x: Union[Variable, NDArray]) -> Variable:
    return Log()(x)

# =============================================================================
# get item
# =============================================================================
class GetItem(Function):
    def __init__(self, key: (
        slice | int | tuple[slice | int] | tuple[slice | int, ...]
    )):
        self.key = key
        
    def forward(self, x: NDArray) -> NDArray:
        return x[self.key]
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return GetItemGrad(self.key, x.shape)(gy)

def get_item(x: Variable, key: (
    slice | int | tuple[slice | int] | tuple[slice | int, ...]
)) -> Variable:
    return GetItem(key)(x)

class GetItemGrad(Function):
    def __init__(self, key: (
        slice | int | tuple[slice | int] | tuple[slice | int, ...]
    ), in_shape: tuple[int, ...]):
        self.key = key
        self.in_shape = in_shape
        
    def forward(self, gy: NDArray) -> NDArray:
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        
        if xp is np:
            np.add.at(gx, self.key, gy)
        else:
            xp.scatter_add(gx, self.key, gy)
        return gx
    
    def backward(self, ggx: Variable) -> Variable:
        return get_item(ggx, self.key)


# =============================================================================
# min max clip
# =============================================================================
class Max(Function):
    def __init__(self, axis: int = None, keepdims: bool = False):
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: NDArray) -> NDArray:
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        y = self.outputs[0]()
        
        shape = utils.max_backward_shape(x, self.axis)
        gy = gy.reshape(shape)
        y = y.reshape(shape)
        cond = (x.data == y.data)
        gy = F.broadcast_to(gy, cond.shape)
        return gy * cond

def max(x: Union[Variable, NDArray], axis: int = None, keepdims: bool = False) -> Variable:
    return Max(axis, keepdims)(x)

class Min(Max):
    def forward(self, x: NDArray) -> NDArray:
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y

def min(x: Union[Variable, NDArray], axis: int = None, keepdims: bool = False) -> Variable:
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min: Union[float, int], x_max: Union[float, int]):
        # TODO expand to support NDArray
        self.x_min = x_min
        self.x_max = x_max
    
    def forward(self, x: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        return xp.clip(x, self.x_min, self.x_max)
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        mask = (x.data >= self.x_min) & (x.data <= self.x_max)
        return gy * mask
    
def clip(x: Union[Variable, NDArray], x_min: Union[float, int], x_max: Union[float, int]) -> Variable:
    return Clip(x_min, x_max)(x)


def accuracy(y: Union[Variable, NDArray], t: Union[Variable, NDArray]) -> Variable:
    """
    This function is not differentiable.
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))