import numpy as np
import sys
import os
import unittest
import timeit
import leaf.nn as nn
from tqdm import tqdm, trange
from leaf import Tensor
from leaf.datautil import fetch_mnist
from leaf.datautil import DataLoader
from leaf.optimizer import SGD, Adam
from leaf.criterion import NLLLoss
np.random.seed(1)
optimizers = {'sgd': SGD, 'adam': Adam}

def _reshape_collate_fn(*tupl):
    X, y, = tupl
    
    if len(X.shape) > 2:
        return Tensor(X.reshape((-1, 784))), Tensor(y)

    return Tensor(X), Tensor(y)

def _conv_collate_fn(*tupl):
    X, y, = tupl
    return Tensor(np.expand_dims(X, 1)), Tensor(y)

class LinearNet(nn.Module):
    def __init__(self, bias):
        self.weights = nn.Sequential(
                nn.Linear(784, 128, bias=bias),
                nn.ReLU(),
                nn.Linear(128, 10, bias=bias),
                nn.LogSoftmax()
                )

    def forward(self, x):
        return self.weights(x)

class ConvNet(nn.Module):
    def __init__(self):
        self.convs = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=(5, 5), stride=1, padding=0),
                nn.MaxPool2d((2, 2)),
                nn.ReLU(),
                nn.Conv2d(4, 8, kernel_size=(3, 3), stride=1, padding=0),
                nn.MaxPool2d((2, 2)),
                nn.ReLU()
                )
        self.dense = nn.Sequential(
                nn.Linear(200, 10, bias=True),
                nn.LogSoftmax()
                )

    def forward(self, x):
        features = self.convs(x)
        features = features.reshape((features.shape[0], -1))
        return self.dense(features)

class TestMNIST(unittest.TestCase):
    def test_linear_dataloader(self):
        def _test_linear_dataloader(optim):
            model = LinearNet(False)
            criterion = NLLLoss()
            optimizer = optimizers[optim](model.parameters(), lr=1e-3)
            training, testing = fetch_mnist(remove_after=False)
            trainloader = DataLoader(
                    training, 
                    batch_size=128, 
                    shuffle=True,
                    collate_fn=_reshape_collate_fn
                )

            for (samples, labels) in (t := tqdm(trainloader)):
                logits = model(samples)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = np.argmax(logits.data, axis=-1)
                acc = (preds == labels.data.astype(int)).mean()

                t.set_description(
                f'{optim} loss {loss.data[0][0]:.3f}  accuracy {acc:.3f}')

            X_test, Y_test = testing.samples, testing.labels
            Y_test_preds_out = model(Tensor(X_test.reshape((-1, 784)))).data
            Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
            acc = (Y_test_preds == Y_test).mean()

            assert acc > 0.9
            sys.stdout.write(f'optimizer {optim} with dataloader got {100*acc:.1f} % acc\n')

        _test_linear_dataloader('sgd')
        _test_linear_dataloader('adam')
    
    def test_linear_boilerplate(self):
        def _test_linear_boilerplate(optim):
            model = LinearNet(True)
            criterion = NLLLoss()
            optimizer = optimizers[optim](model.parameters(), lr=1e-3)
            training, testing = fetch_mnist(remove_after=False)
            X_train, Y_train = training.samples, training.labels
            X_test, Y_test = testing.samples, testing.labels
            X_train = X_train.reshape((-1, 784))
            X_test = X_test.reshape((-1, 784))
            batch_size = 128

            for _ in (t := trange(400, disable=os.getenv('CI') is not None)):
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
                f'{optim} loss {loss.data[0][0]:.3f}  accuracy {acc:.3f}')

            Y_test_preds_out = model(Tensor(X_test.reshape((-1, 784)))).data
            Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
            acc = (Y_test_preds == Y_test).mean()

            assert acc > 0.9
            sys.stdout.write(f'optimizer {optim} with boilerplate got {100*acc:.1f} % acc\n')

        _test_linear_boilerplate('sgd')
        _test_linear_boilerplate('adam')

    def test_conv2d_dataloader(self):
        def _test_conv2d_dataloader(optim):
            model = ConvNet()
            criterion = NLLLoss()
            optimizer = optimizers[optim](model.parameters(), lr=1e-3)
            training, testing = fetch_mnist(remove_after=False)
            trainloader = DataLoader(
                    training,
                    batch_size=128,
                    shuffle=True,
                    collate_fn=_conv_collate_fn
                    )

            for (samples, labels) in (t := tqdm(trainloader)):
                logits = model(samples)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = np.argmax(logits.data, axis=-1)
                acc = (preds == labels.data.astype(int)).mean()

                t.set_description(
                f'{optim} loss {loss.data[0][0]:.3f}  accuracy {acc:.3f}')

            X_test, Y_test = testing.samples, testing.labels
            Y_test_preds_out = model(Tensor(X_test.reshape((-1, 1, 28, 28)))).data
            Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
            acc = (Y_test_preds == Y_test).mean()

            assert acc > 0.9
            sys.stdout.write(f'optimizer {optim} with dataloader got {100*acc:.1f} % acc\n')

        _test_conv2d_dataloader('sgd')
        _test_conv2d_dataloader('adam')

