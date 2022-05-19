""" ---------------------------------------------------------------------------
Initialize the leaf, minimalistic deep learning framework, module and all its 
submodules and dependencies. Register all available ops from `leaf/functions/`
as attributes for the Tensor class.

Authors: Wilhelm Ågren <wilhelmagren98@gmail.com>
Last edited: 19-05-2022
License: Apache 2.0
--------------------------------------------------------------------------- """
from .tensor import Tensor, concatenate
from .trainer import Trainer
from leaf import optimizer
from leaf import criterion
from leaf import nn
from leaf import initializer
from leaf import datautil

# register ops for Tensor below...
import sys
import os
import inspect
import importlib
from functools import partialmethod

def _register_from_import(namespace):
    for name, func in inspect.getmembers(namespace, inspect.isclass):
        setattr(Tensor, name.lower(), partialmethod(func.apply, func))

opfiles = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'functions'))
opfiles = [f.split('.')[0] for f in opfiles if f.endswith('_ops.py')]
for optype in opfiles:
    try:
        _register_from_import(importlib.import_module('leaf.functions.'+optype))
    except ImportError:
        sys.stderr.write(f'could not import module <leaf.functions.{optype}>')

