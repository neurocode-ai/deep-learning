""" ---------------------------------------------------------------------------
Linear transform, layer, implementation utilizing `matmul` processing op
and `add` binary op. Then number of parameters for a linear layer is 
calculated as `num_params = (fan_in + 1)* fan_out` if bias=True.

Authors: Wilhelm Ågren <wilhelmagren98@gmail.com>
Last edited: 19-05-2022
License: Apache 2.0
--------------------------------------------------------------------------- """
from leaf import Tensor
from .nn import Module

class Linear(Module):
    """ linear layer implementation, inheriting parent class Module.
    This Module requires input to be a 2D Tensor that has the matmul
    operator implemented. 

    Parameters
    ----------
    fan_in: int
        Dimensionality of input, i.e., the number of input features.
    fan_out: int
        Wanted dimensionality of output, in second axis of Tensor.
    bias: bool
        Toggle whether or not to use bias parameter in the layer, 
        if bias=True, then the bias is added to the output of the 
        matmul operator.

    """
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = Tensor.uniform(fan_in, fan_out, requires_grad=True)
        self.bias = Tensor.uniform(fan_out, requires_grad=True) if bias else None

    def forward(self, x):
        """ propagate input through the linear layer.
        Basically perform the linear transform operation,        

        Parameters
        ----------
        Input   x: (batch_size, fan_in)
        Weight  w: (fan_in, fan_out)
        Bias    b: (fan_out, )

        Output = x @ w + b
        (batch_size, fan_out) = (batch_size, fan_in) @ (fan_in, fan_out) + (fan_out, )

        """
        matmul = x.matmul(self.weight)

        if self.bias is None:
            return matmul
        
        return matmul.add(self.bias)

