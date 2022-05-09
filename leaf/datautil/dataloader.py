import numpy as np
from leaf import Tensor
from .dataset import Dataset

class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, allocate_fn=None):
        assert batch_size > 0
        assert isinstance(dataset, Dataset)
    
        if allocate_fn is None:
            allocate_fn = _default_allocate_fn
        
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._allocate_fn = allocate_fn

    def __len__(self):
        return len(self._dataset) // self._batch_size

    def __iter__(self):
        for _ in range(len(self)):
            indices = np.random.choice(len(self._dataset), self._batch_size)
            samples, labels, = self._dataset[indices]
            yield self._allocate_fn(samples), self._allocate_fn(labels)

def _default_allocate_fn(items):
    return Tensor(items)

