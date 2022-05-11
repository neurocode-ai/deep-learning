from .tensor import Tensor
from .trainer import Trainer
from leaf import optimizer
from leaf import criterion
from leaf import nn
from leaf import initializer
from leaf import datautil


# Register all Function ops to the Tensor here!!!
import sys
import os
import inspect
import importlib
from functools import partialmethod

def _register_op(name, func):
    setattr(Tensor, name, partialmethod(func.apply, func))

def _register_from_import(namespace):
    for name, cls in inspect.getmembers(namespace, inspect.isclass):
        _register_op(name.lower(), cls)

opfiles = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'functions'))
opfiles = [f.split('.')[0] for f in opfiles if f.endswith('_ops.py')]
for optype in opfiles:
    try:
        _register_from_import(importlib.import_module('leaf.functions.'+optype))
    except ImportError:
        sys.stderr.write(f'could not import module <leaf.functions.{optype}>')

