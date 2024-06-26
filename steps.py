from core.core import *

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

x = Variable(np.array(0.5))
y = square(exp(square(x)))

y.backward()
print(x.grad)