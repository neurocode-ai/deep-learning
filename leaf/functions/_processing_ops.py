from .function import Function

class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, grad, **kwargs):
        x, y, = self.saved_tensors
        return grad @ y.T, x.T @ grad

class Dot(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.dot(y)
    
    def backward(self, grad, **kwargs):
        x, y, = self.saved_tensors
        return grad.dot(y.T), x.T.dot(grad)

