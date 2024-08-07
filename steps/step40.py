if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from my_framework.utils import sum_to
from my_framework import Variable

# x0 = np.array([[1, 2, 3], [4, 5, 6]])
# y = sum_to(x0, (1, 3))
# print(y)

# x0 = Variable(np.array([1, 2, 3]))
# x1 = Variable(np.array([10]))

# y = x0 + x1
# print(y)

# y.backward()
# print(x0.grad, x1.grad)

x0 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
x1 = x0.reshape((6,))
print(x1)