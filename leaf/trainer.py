import leaf.nn as nn
from leaf.optimizer import Optimizer
from leaf.criterion import Criterion

class Trainer(object):
    def __init__(self, dataloader, test=None, max_epochs=100, **kwargs):
        assert max_epochs > 0
        self._dataloader = dataloader
        self._max_epochs = max_epochs
        
    def fit(self, model, optimizer, criterion):
        assert isinstance(model, nn.Module)
        assert isinstance(optimizer, Optimizer)
        assert isinstance(criterion, Criterion)

        # TODO: implement model training here ...

