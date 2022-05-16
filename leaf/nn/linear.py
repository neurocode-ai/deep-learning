from leaf import Tensor
from .nn import Module

class Linear(Module):
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = Tensor.uniform(fan_in, fan_out, requires_grad=True)
        self.bias = Tensor.uniform(fan_out, requires_grad=True) if bias else None

    def forward(self, x):
        matmul = x.matmul(self.weight)

        if self.bias is None:
            return matmul
        
        return matmul.add(self.bias)

