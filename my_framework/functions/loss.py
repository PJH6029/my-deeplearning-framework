from typing import Optional, Union
import numpy as np

import my_framework.core.base as base
from my_framework.types import NDArray
import my_framework.functions as F
from my_framework import utils

def mean_squared_error_naive(x0, x1):
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)

class MeanSquaredError(base.Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        diff = x0 - x1
        y = (diff ** 2).mean()
        return y
        
    def backward(self, gy: base.Variable) -> tuple[base.Variable, base.Variable]:
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0: Union[base.Variable, NDArray], x1: Union[base.Variable, NDArray]) -> base.Variable:
    return MeanSquaredError()(x0, x1)


def softmax_cross_entropy_naive(x: Union[base.Variable, NDArray], t: Union[base.Variable, NDArray]) -> base.Variable:
    x, t = base.as_variable(x), base.as_variable(t)
    N = x.shape[0]

    # p = F.softmax(x)
    # p = F.clip(p, 1e-15, 1.0)
    # log_p = F.log(p)
    log_p = F.log_softmax(x)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * F.sum(tlog_p) / N
    return y

class SoftmaxCrossEntropy(base.Function):
    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        N = x.shape[0]
        # log_p = F.log_softmax(x)
        
        # Cannot use my_framework.functions, since it returns Variable, not NDArray
        log_z = utils.logsumexp(x, axis=1) 
        log_p = x - log_z
        log_p = log_p[np.arange(N), t]
        y = -log_p.sum() / np.float32(N)
        return y
    
    def backward(self, gy: base.Variable) -> base.Variable:
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = F.softmax(x)
        
        # convert to one-hot
        t_one_hot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        gx = (y - t_one_hot) * gy
        return gx
    
def softmax_cross_entropy(x: Union[base.Variable, NDArray], t: Union[base.Variable, NDArray]) -> base.Variable:
    return SoftmaxCrossEntropy()(x, t)
