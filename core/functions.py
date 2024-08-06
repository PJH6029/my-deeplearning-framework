from core.types import *
from core.core_simple import Variable, Function, as_variable
from core import utils
import core.cuda

class SumTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        
    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy: NDArray) -> NDArray:
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class BroadcastTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
    
    def forward(self, x: NDArray) -> NDArray:
        self.x_shape = x.shape
        xp = core.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy: NDArray) -> NDArray:
        gx = sum_to(gy, self.x_shape) # TODO type unmatch
        return gx

def broadcast_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

# class GetItem(Function):
#     def __init__(self, slices: tuple[slice, ...]) -> None:
#         self.slices = slices TODO

# def get_item(x: Variable, slices: tuple[slice, ...]) -> Variable:
#     # y = x.data[slices]
#     # return as_variable(y) # TODO
#     return