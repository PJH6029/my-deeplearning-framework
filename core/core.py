import numpy as np
import heapq, itertools

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # print(id(self), id(self.creator.outputs[0])) # same id for first output
        
        funcs = []
        seen_funcs = set()
        entry_count = itertools.count()
        
        def add_func(f):
            # avoid inserting duplicate functions
            if f not in seen_funcs:
                # generation is used to sort functions in topological order
                # if f.generation is same, heapq will sort them based on the order of insertion
                # sorting graph topologically every time is costly, so we use heapq
                priority = -f.generation
                ecount = next(entry_count)
                heapq.heappush(funcs, (priority, ecount, f))
                seen_funcs.add(f)
        
        def pop_func():
            _, _, f = heapq.heappop(funcs) # fetch function with max generation
            return f
        
        add_func(self.creator)
        while funcs:
            f = pop_func()
            grad_ys = [output.grad for output in f.outputs]
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
    
    def clear_grad(self):
        self.grad = None
        
class Function:
    def __call__(self, *inputs):
        x_data = [x.data for x in inputs]
        y_data = self.forward(*x_data)
        if not isinstance(y_data, tuple):
            y_data = (y_data,)
        
        outputs = [Variable(as_array(y)) for y in y_data]
        
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
    

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x