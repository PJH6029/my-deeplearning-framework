from typing import Any, Optional, Union
import numpy as np
from numpy.typing import NDArray, DTypeLike
import heapq, itertools
import weakref
import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name: str, value: Any):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    __array_priority__ = 200
    
    def __init__(self, data: Optional[Any], name: Optional[str] = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        
        self.data: NDArray = data
        self.name: Optional[str] = name
        self.grad: Optional[NDArray] = None
        self.creator: Function = None
        self.generation: int = 0
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def set_creator(self, func: "Function"):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self, retain_grad: bool = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # print(id(self), id(self.creator.outputs[0])) # same id for first output
        
        funcs: list[Function] = []
        seen_funcs: set[Function] = set()
        entry_count = itertools.count()
        
        def add_func(f: Function):
            # avoid inserting duplicate functions
            if f not in seen_funcs:
                # generation is used to sort functions in topological order
                # if f.generation is same, heapq will sort them based on the order of insertion
                # sorting graph topologically every time is costly, so we use heapq
                priority = -f.generation
                ecount = next(entry_count)
                heapq.heappush(funcs, (priority, ecount, f))
                seen_funcs.add(f)
        
        def pop_func() -> Function:
            _, _, f = heapq.heappop(funcs) # fetch function with max generation
            return f
        
        add_func(self.creator)
        while funcs:
            f = pop_func()
            grad_ys = [output().grad for output in f.outputs]
            grad_xs = f.backward(*grad_ys)
            if not isinstance(grad_xs, tuple):
                grad_xs = (grad_xs,)
            
            for x, grad_x in zip(f.inputs, grad_xs):
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad = x.grad + grad_x # Add gradient from different paths
                    
                    # if they are same reference, in-place operation will make unexpected result
                    # if grad_x refers to the same object as self.grad, self.grad will also refer x.grad
                    # see deep-learning-from-scratch-3 appendix A (page 525)
                    # x.grad += grad_x 
                
                if x.creator is not None:
                    add_func(x.creator)
                    
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y: weakref.ref
    
    def clear_grad(self):
        self.grad = None

        
def as_array(x: Any) -> NDArray:
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj: Any) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs: Any) -> Union[Variable, list[Variable]]:
        _inputs = [as_variable(x) for x in inputs]
        
        xs = [x.data for x in _inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in _inputs])
            
            for output in outputs:
                output.set_creator(self)
            
            self.inputs = _inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: NDArray) -> tuple[NDArray, ...] | NDArray:
        raise NotImplementedError()

    def backward(self, *gys: NDArray) -> tuple[NDArray, ...] | NDArray:
        raise NotImplementedError()


# overloading operators
# TODO type hinting for operators
class Add(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        return x0 + x1
    
    def backward(self, gy: NDArray) -> tuple[NDArray, NDArray]:
        gx0, gx1 = gy, gy
        return gx0, gx1

def add(x0: Variable, x1: Any) -> Variable: # x0 goes to self
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        y = x0 * x1
        return y

    def backward(self, gy: NDArray) -> tuple[NDArray, NDArray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy * x1
        gx1 = gy * x0
        return gx0, gx1

def mul(x0: Variable, x1: Any) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x: NDArray) -> NDArray:
        return -x

    def backward(self, gy: NDArray) -> NDArray:
        return -gy
    
def neg(x: Variable) -> Variable:
    return Neg()(x)


class Sub(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        return x0 - x1
    
    def backward(self, gy: NDArray) -> tuple[NDArray, NDArray]:
        gx0, gx1 = gy, -gy
        return gx0, gx1

def sub(x0: Variable, x1: Any) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0: Variable, x1: Any) -> Variable:
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        return x0 / x1
    
    def backward(self, gy: NDArray) -> tuple[NDArray, NDArray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0: Variable, x1: Any) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0: Variable, x1: Any) -> Variable:
    x1 = as_array(x1)
    return div(x1, x0)

class Pow(Function):
    def __init__(self, c: float) -> None:
        self.c = c
    
    def forward(self, x: NDArray) -> NDArray:
        return x ** self.c
    
    def backward(self, gy: NDArray) -> NDArray:
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x: Variable, c: float) -> Variable:
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
