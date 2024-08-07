if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from my_framework.utils import sum_to
from my_framework import Variable
from my_framework import functions as F

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
print(y)

z = y.T @ x
print(z.shape)

y.backward()
print(x.grad)
print(W.grad)
