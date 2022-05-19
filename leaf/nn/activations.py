""" ---------------------------------------------------------------------------
Activation functions that can be used when creating user defined nn.Module
or nn.Sequence, basically only wraps the specific Function, but streamlines
the creation of neural networks.

Authors: Wilhelm Ågren <wilhelmagren98@gmail.com>
Last edited: 19-05-2022
License: Apache 2.0
--------------------------------------------------------------------------- """

from .nn import Module

class ReLU(Module):
    """ look into doing this in-place... """
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

class Tanh(Module):
    def forward(self, x):
        return x.tanh()

class LogSoftmax(Module):
    def forward(self, x):
        return x.logsoftmax()

