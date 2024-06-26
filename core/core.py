import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # print(id(self), id(self.creator.outputs[0])) # same id for first output
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
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
                    funcs.append(x.creator)
        
class Function:
    def __call__(self, *inputs):
        x_data = [x.data for x in inputs]
        y_data = self.forward(*x_data)
        if not isinstance(y_data, tuple):
            y_data = (y_data,)
        
        outputs = [Variable(as_array(y)) for y in y_data]
        
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