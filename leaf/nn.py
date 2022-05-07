from leaf import Tensor

class Module(object):
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError(
        f'user defined nn.Module has not implemented forward pass')

    def parameters(self):
        # TODO: implement this
        return []


class Dense(Module):
    def __init__(self, fan_in, fan_out, use_bias=True):
        self.weight_ = Tensor.uniform(fan_in, fan_out)
        self.bias_ = Tensor.uniform(fan_out) if use_bias else None

    def forward(self, x):
        result = x.dot(self.weight_)
        
        if self.use_bias:
            return result.add(self.bias_)

        return result
