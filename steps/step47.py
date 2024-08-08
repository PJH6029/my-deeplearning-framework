if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from my_framework import Variable

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))

print(x[0])