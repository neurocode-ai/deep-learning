import numpy as np
import os
import unittest
import timeit
import leaf.nn as nn
from tqdm import trange
from leaf import Tensor
from leaf.datautil import fetch_mnist
from leaf.datautil import DataLoader
from leaf.optimizer import SGD, Adam
from leaf.criterion import NLLLoss

def _reshape_collate_fn(t):
    if len(t.shape) > 2:
        return Tensor(t.reshape((-1, 784)))

    return Tensor(t)

class LinearNet(nn.Module):
    def __init__(self, bias):
        self.weights = nn.Sequential(
                nn.Linear(784, 128, use_bias=bias),
                nn.ReLU(),
                nn.Linear(128, 10, use_bias=bias),
                nn.LogSoftmax()
                )

    def forward(self, x):
        return self.weights(x)

class TestMNIST(unittest.TestCase):
    def test_linear_SGD_dataloader(self):
        model = LinearNet(False)
        criterion = NLLLoss()
        optimizer = SGD(model.parameters(), lr=1e-3)
        training, testing = fetch_mnist(remove_after=True)
        trainloader = DataLoader(
                training, 
                batch_size=128, 
                shuffle=True,
                collate_fn=_reshape_collate_fn
            )

        for _ in (t := trange(2, disable=os.getenv('CI') is not None)):
            for samples, labels in trainloader:
                logits = model(samples)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = np.argmax(logits.data, axis=-1)
                acc = (preds == labels.data.astype(int)).mean()

                t.set_description(
                f'loss {loss.data[0][0]:.3f}  accuracy {acc:.3f}')
    
    def test_linear_SGD_boilerplate(self):
        model = LinearNet(True)
        criterion = NLLLoss()
        optimizer = SGD(model.parameters(), lr=1e-3)
        training, testing = fetch_mnist(remove_after=True)
        X_train, Y_train = training._samples, training._labels
        X_test, Y_test = testing._samples, testing._labels
        X_train = X_train.reshape((-1, 784))
        X_test = X_test.reshape((-1, 784))
        batch_size = 128

        for _ in (t := trange(300, disable=os.getenv('CI') is not None)):
            indices = np.random.randint(0, X_train.shape[0], size=(batch_size))
            samples = Tensor(X_train[indices])
            targets = Tensor(Y_train[indices])

            logits = model(samples)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = np.argmax(logits.data, axis=-1)
            acc = (preds == targets.data.astype(int)).mean()

            t.set_description(
            f'loss {loss.data[0][0]:.3f}  accuracy {acc:.3f}')

