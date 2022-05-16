from leaf import Tensor
from .nn import Module

class Conv2d(Module):
    def __init__(self, in_C, out_C, kernel_size, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        k_H, k_W = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = Tensor.uniform(out_C, in_C, k_H, k_W, requires_grad=True)

    def forward(self, x):
        return x.conv2d(
                self.kernel, 
                stride=self.stride, 
                padding=self.padding
                )

