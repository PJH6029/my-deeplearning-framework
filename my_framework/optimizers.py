from typing import Optional, Callable
import math
import numpy as np

from my_framework import Layer, Parameter
from my_framework.types import NDArray

class Optimizer:
    def __init__(self):
        self.target: Optional[Layer] = None
        self.hooks: list[Callable[[list[Parameter]], None]] = []
        
    def setup(self, target: Layer) -> 'Optimizer':
        self.target = target
        return self
    
    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        # preprocess
        for f in self.hooks:
            f(params)
        
        for param in params:
            self.update_one(param)
    
    def update_one(self, param: Parameter):
        raise NotImplementedError()
    
    def add_hook(self, f: Callable[[list[Parameter]], None]):
        self.hooks.append(f)
        
class SGD(Optimizer):
    def __init__(self, lr: float = 1e-2):
        super().__init__()
        self.lr = lr
    
    def update_one(self, param: Parameter):
        param.data -= self.lr * param.grad.data
        

class MomentumSGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs: dict[Parameter, NDArray] = {}
    
    def update_one(self, param: Parameter):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, lr: float = 0.001, eps: float = 1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs: dict[Parameter, NDArray] = {}
    
    def update_one(self, param: Parameter):
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)
        
        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]
        
        h += grad * grad
        param.data -= lr * grad / (np.sqrt(h) + eps)


class AdaDelta(Optimizer):
    def __init__(self, rho: float = 0.95, eps: float = 1e-6):
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msgs: dict[Parameter, NDArray] = {}
        self.msdxs: dict[Parameter, NDArray] = {}
    
    def update_one(self, param: Parameter):
        key = id(param)
        if key not in self.msgs:
            self.msgs[key] = np.zeros_like(param.data)
            self.msdxs[key] = np.zeros_like(param.data)
            
        msg, msdx = self.msgs[key], self.msdxs[key]
        rho = self.rho
        eps = self.eps
        grad = param.grad.data
        
        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = np.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx


class Adam(Optimizer):
    def __init__(self, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms: dict[Parameter, NDArray] = {}
        self.vs: dict[Parameter, NDArray] = {}
        self.t = 0
    
    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)
    
    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param: Parameter):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)
        
        m, v = self.ms[key], self.vs[key]
        beta1, beta2 = self.beta1, self.beta2
        eps = self.eps
        grad = param.grad.data
        
        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (np.sqrt(v) + eps)
