if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core import Variable, Function

print(Variable.__add__)
x = Variable(np.array(1.0))
y = Variable(np.array(2.0))
z = 2 * x + y
w = 2 * z
w.backward()

# print(y)
print(x.grad)
print(y.grad)
print(z.grad)
print(w.grad)

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def gx_2(x):
    return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    
    y = f(x)
    x.clear_grad()
    y.backward(create_graph=True)
    gx = x.grad
    
    x.clear_grad()
    gx.backward()
    gx2 = x.grad
    
    x.data -= gx.data / gx2.data
    