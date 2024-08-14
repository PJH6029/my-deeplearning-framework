import weakref, os
from typing import Any, Generator, Optional
import numpy as np

from my_framework.core import Parameter, Variable
import my_framework.functions as F

class Layer:
    def __init__(self) -> None:
        self._params: set[str] = set()
    
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)
    
    def __call__(self, *inputs: Variable) -> Variable: # TODO type
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs: Variable) -> Variable:
        raise NotImplementedError()
    
    def params(self) -> Generator[Parameter, None, None]:
        for name in self._params:
            obj = self.__dict__[name]
            
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj
        
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
            
    def _flatten_params(self, params_dict: dict[str, Variable], parent_key: str = "", sep: str = "/") -> None:
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + sep + name if parent_key else name
            
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key, sep)
            else:
                params_dict[key] = obj
    
    def save_params(self, path: str) -> None:
        self.to_cpu()
        
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None and param.data is not None}
        
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_params(self, path: str) -> None:
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
    
    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
    
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()


class Linear(Layer):
    def __init__(self, out_size: int, bias: bool = True, dtype: np.dtype = np.float32, in_size: Optional[int] = None) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()
        
        if bias:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
        else:
            self.b = None
    
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    
    def forward(self, x: Variable) -> Variable:
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        
        y = F.linear(x, self.W, self.b)
        return y
