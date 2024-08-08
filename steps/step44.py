if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from matplotlib import pyplot as plt

from my_framework.utils import sum_to
from my_framework import Variable, Parameter
import my_framework.functions as F
import my_framework.layers as L

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    h = F.sigmoid(l1(x))
    y = l2(h)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)

plt.scatter(x, y, s=10)

plt.xlabel('x')
plt.ylabel('y')

t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)

plt.plot(t, y_pred.data, color='r')
plt.show()