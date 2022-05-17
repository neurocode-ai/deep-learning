from leaf import Tensor
from .nn import Module

class BatchNorm2d(Module):
    """ https://arxiv.org/abs/1502.03167 """
    def __init__(self, features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        self.weight = Tensor.ones(features, requires_grad=True)
        self.bias = Tensor.zeros(features, requires_grad=True)
        self.running_mean = Tensor.zeros(features, requires_grad=False)
        self.running_var = Tensor.ones(features, requires_grad=False)

    def forward(self, x, training=True):
        """ learnable parameters  """

        if training:
            x_detached = x.detach()
            batch_mean = x.mean(axis=(0, 2, 3), keepdims=False)
            y = x_detached - batch_mean.reshape(shape=(1, -1, 1, 1))
            batch_var = (y * y).mean(axis=(0, 2, 3), keepdims=False)

        inv_moment_mean = self.running_mean.mul(1 - self.momentum)
        moment_mean = batch_mean.mul(self.momentum)
        self.running_mean = inv_moment_mean + moment_mean

        inv_moment_var = self.running_var.mul(1 - self.momentum)
        moment_var = batch_var.mul(self.momentum)
        self.running_var = inv_moment_var + moment_var

        # "The normalization of activations that depends on the mini-batch
        #  allows efficient training, but is neither necessary nor desirable
        #  during inference; we want the output to depend only on the input,
        #  deterministically." (Ioffe and Szegedy, 2015)
        # so we use the population statistics when performing inference.
        if training:
            return self.normalize(x, batch_mean, batch_var)
        
        return self.normalize(x, self.running_mean, self.running_var)

    def normalize(self, x, mean, var):
        x = (x - mean.reshape(shape=(1, -1, 1, 1))) * self.weight.reshape(shape=(1, -1, 1, 1))
        x = x.div(var.add(self.eps).reshape(shape=(1, -1, 1, 1)).pow(Tensor(0.5))) + self.bias.reshape(shape=(1, -1, 1, 1))
        return x

