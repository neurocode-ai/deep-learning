""" ---------------------------------------------------------------------------
Neural network module parent class Module which is inherited for all buildin 
blocks. The Sequential class is used to streamline defining a neural network,
taking a number of Module objects which are used with a single forward call.

See class specific documentation for more information.

Authors: Wilhelm Ågren <wilhelmagren98@gmail.com>
Last edited: 19-05-2022
License: Apache 2.0
--------------------------------------------------------------------------- """
import leaf
from leaf import Tensor

class Module(object):
    """ parent class for neural network building blocks, i.e Modules, that
    defines invoking structure. The neural network Module specific
    forward pass is called by invoking __call__ on the building block
    that inherits this class. Because this is just a parent class defining
    behavior for subsequent neural network building blocks, no __init__
    method is defined and can not be inherited by subclasses.

    """
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError(
        f'user defined nn.Module has not implemented forward pass')

    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                params.extend([attr] if attr.requires_grad else [])
            if isinstance(attr, list):
                params.extend([p for p in attr if p.requires_grad])
            if isinstance(attr, (Module, Sequential)):
                # recursively find all Tensor parameters if nested Module,
                # i.e. neural network building blocks in Sequential or
                # parent Module.
                params.extend([p for p in attr.parameters() if p.requires_grad])
        return params

class Sequential(object):
    """ wrapper implementation for multiple Module objects to
    simplify forward pass definition in user defined neural network.

    Parameters
    ----------
    *modules: tuple | Module
        Colection of positional arguments to use with propagation
        of forwards pass call. If it is a tuple, all members
        should be Module objects, otherwise they won't be callable.

    """
    def __init__(self, *modules):
        assert all(isinstance(m, Module) for m in modules)
        self.modules = modules

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x
    
    def parameters(self):
        params = []
        for m in self.modules:
            params.extend(m.parameters())
        return params


