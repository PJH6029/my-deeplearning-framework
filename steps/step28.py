if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from my_framework import Variable, Function

def rosenbrock(x0: Variable, x1: Variable) -> Variable:
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 1e-3
num_iters = 10000

for i in range(num_iters):
    print(x0, x1)
    
    y = rosenbrock(x0, x1)
    
    x0.clear_grad()
    x1.clear_grad()
    y.backward()
    
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad