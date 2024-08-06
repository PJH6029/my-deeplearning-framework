if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
from core import Function, Variable
from core.types import *
from core import utils

class Sin(Function):
    def forward(self, x: NDArray) -> NDArray:
        return np.sin(x)
    
    def backward(self, gy: NDArray) -> NDArray:
        x = self.inputs[0].data
        return gy * np.cos(x)

def sin(x: Variable) -> Variable:
    return Sin()(x)

x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward()
print(y.data)
print(x.grad)

def my_sin(x: Variable, threshold=1e-4) -> Variable:
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

x.clear_grad()
y.clear_grad()
y = my_sin(x)
y.backward()
print(y.data)
print(x.grad)
utils.plot_dot_graph(y, verbose=False, to_file="my_sin.png")

x.clear_grad()
y.clear_grad()
y = my_sin(x, threshold=1e-150)
y.backward()
utils.plot_dot_graph(y, verbose=False, to_file="my_sin_1e_150.png")