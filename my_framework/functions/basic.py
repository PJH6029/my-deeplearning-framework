from typing import Union
import numpy as np

from my_framework.core import Variable, Function
from my_framework.types import NDArray

class Sin(Function):
    def forward(self, x: NDArray) -> NDArray:
        y = np.sin(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * cos(x)
        return gx
    
def sin(x: Union[Variable, NDArray]) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x: NDArray) -> NDArray:
        y = np.cos(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x: Union[Variable, NDArray]) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: NDArray) -> NDArray:
        y = np.tanh(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx
    
def tanh(x: Union[Variable, NDArray]) -> Variable:
    return Tanh()(x)


class Exp(Function):
    def forward(self, x: NDArray) -> NDArray:
        y = np.exp(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        return gy * y

def exp(x: Union[Variable, NDArray]) -> Variable:
    return Exp()(x)


class Log(Function):
    def forward(self, x: NDArray) -> NDArray:
        y = np.log(x)
        return y
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return gy / x
    
def log(x: Union[Variable, NDArray]) -> Variable:
    return Log()(x)