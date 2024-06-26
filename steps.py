from core.core import *

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

x = Variable(np.array(0.5))
y = square(exp(square(x)))

y.backward()
print(x.grad)

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))

c = add(square(a), square(b))
c.backward()
print(c.data)
print(a.grad, b.grad)

x = Variable(np.array(3))
y = add(x, x)
print("y data, grad", y.data, y.grad)
y.backward()
print(y.grad, id(y.grad))
print(x.grad, id(x.grad))

x = Variable(np.array(3.0))
y = add(add(x, x), x)
y.backward()
print(x.grad)

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))   
y.backward()
print(y.data)
print(x.grad)