from typing import Any, Optional, Union, Type
import numpy as np
import heapq, itertools
import weakref

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray,)

from my_framework.types import NDArray
from my_framework.core.config import Config, using_config
import my_framework

# =============================================================================
# Utility functions
# =============================================================================
def as_array(x: Any, array_module=np) -> NDArray:
    if np.isscalar(x):
        return array_module.array(x)
    return x

def as_variable(obj: Union["Variable", NDArray]) -> "Variable":
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

# =============================================================================
# Base Function Class
# =============================================================================
class Function:
    def __call__(self, *inputs: Any) -> Union["Variable", list["Variable"]]:
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

    def backward(self, *gys: "Variable") -> Union["Variable", tuple["Variable", ...]]:
        raise NotImplementedError()

# =============================================================================
# Base Variable class
# =============================================================================
class Variable:
    __array_priority__ = 200
    
    def __init__(self, data: Any, name: Optional[str] = None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError(f'{type(data)} is not supported')
        
        self.data: Optional[NDArray] = data
        self.name: Optional[str] = name
        self.grad: Optional["Variable"] = None
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

    def backward(self, retain_grad: bool = False, create_graph: bool = False):    
        if self.creator is None:
            # if self is a leaf variable
            return
        
        if self.grad is None:
            xp = my_framework.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

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
            
            with using_config('enable_backprop', create_graph):
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
    
    def cleargrad(self):
        self.grad = None
        
    def reshape(self, *shape: int) -> "Variable":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return my_framework.functions.reshape(self, shape)
    
    def transpose(self, *axes: int) -> "Variable":
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and (isinstance(axes[0], (tuple, list)) or axes[0] is None):
            axes = axes[0]
        return my_framework.functions.transpose(self, axes)
    
    @property
    def T(self):
        return self.transpose()
    
    def sum(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> "Variable":
        return my_framework.functions.sum(self, axis, keepdims)
    
    def to_cpu(self):
        if self.data is not None:
            self.data = my_framework.cuda.as_numpy(self.data)
    
    def to_gpu(self):
        if self.data is not None:
            self.data = my_framework.cuda.as_cupy(self.data)
    
    # Note: manually overriding operators, to make the result can be type-checked
    def __add__(self, other: Any) -> "Variable":
        return add(self, other)

    def __radd__(self, other: Any) -> "Variable":
        return add(self, other)
    
    def __mul__(self, other: Any) -> "Variable":
        return mul(self, other)
    
    def __rmul__(self, other: Any) -> "Variable":    
        return mul(self, other)
    
    def __neg__(self) -> "Variable":
        return neg(self)
    
    def __sub__(self, other: Any) -> "Variable":
        return sub(self, other)
    
    def __rsub__(self, other: Any) -> "Variable":
        return rsub(self, other)
    
    def __truediv__(self, other: Any) -> "Variable":
        return div(self, other)
    
    def __rtruediv__(self, other: Any) -> "Variable":
        return rdiv(self, other)
    
    def __pow__(self, other: Any) -> "Variable":
        return pow(self, other)

    def __matmul__(self, other: "Variable") -> "Variable":
        return my_framework.functions.matmul(self, other)
    
    def __getitem__(self, key: (
        slice | int | tuple[slice | int] | tuple[slice | int, ...]    
    )) -> "Variable":
        return my_framework.functions.get_item(self, key)


class Parameter(Variable):
    pass

# =============================================================================
# Overriding operators
# =============================================================================
class Add(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        return x0 + x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0, gx1 = gy, gy
        if x0.shape != x1.shape:
            # broadcasted in forward
            # sum along broadcasted axes
            gx0 = my_framework.functions.sum_to(gx0, x0.shape)
            gx1 = my_framework.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def add(x0: Variable, x1: Any) -> Variable:
    if isinstance(x1, Variable):
        return Add()(x0, x1)
    return Add()(x0, as_array(x1, my_framework.cuda.get_array_module(x0.data)))

class Mul(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        return x0 * x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = my_framework.functions.sum_to(gx0, x0.shape)
            gx1 = my_framework.functions.sum_to(gx1, x1.shape)
        return gy * x1, gy * x0

def mul(x0: Variable, x1: Any) -> Variable:
    if isinstance(x1, Variable):
        return Mul()(x0, x1)
    return Mul()(x0, as_array(x1, my_framework.cuda.get_array_module(x0.data)))


class Neg(Function):
    def forward(self, x: NDArray) -> NDArray:
        return -x

    def backward(self, gy: Variable) ->Variable:
        return -gy
    
def neg(x: Variable) -> Variable:
    return Neg()(x)


class Sub(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        return x0 - x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy
        gx1 = -gy
        if x0.shape != x1.shape:
            gx0 = my_framework.functions.sum_to(gx0, x0.shape)
            gx1 = my_framework.functions.sum_to(gx1, x1.shape)
        return gx0, gx1
        
    
def sub(x0: Variable, x1: Any) -> Variable:
    if isinstance(x1, Variable):
        return Sub()(x0, x1)
    return Sub()(x0, as_array(x1, my_framework.cuda.get_array_module(x0.data)))

def rsub(x0: Variable, x1: Any) -> Variable:
    if isinstance(x1, Variable):
        return Sub()(x1, x0)
    return Sub()(as_array(x1, my_framework.cuda.get_array_module(x0.data)), x0)

class Div(Function):
    def forward(self, x0: NDArray, x1: NDArray) -> NDArray:
        return x0 / x1
    
    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            gx0 = my_framework.functions.sum_to(gx0, x0.shape)
            gx1 = my_framework.functions.sum_to(gx1, x1.shape)
        return gx0, gx1
    
def div(x0: Variable, x1: Any) -> Variable:
    if isinstance(x1, Variable):
        return Div()(x0, x1)
    return Div()(x0, as_array(x1, my_framework.cuda.get_array_module(x0.data)))

def rdiv(x0: Variable, x1: Any) -> Variable:
    if isinstance(x1, Variable):
        return Div()(x1, x0)
    return Div()(as_array(x1, my_framework.cuda.get_array_module(x0.data)), x0)


class Pow(Function):
    def __init__(self, c: float):
        self.c = c
        
    def forward(self, x: NDArray) -> NDArray:
        return x ** self.c
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        c = self.c
        return gy * c * x ** (c - 1)
    
def pow(x: Variable, c: float) -> Variable:
    return Pow(c)(x)