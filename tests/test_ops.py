import torch
import torch.nn.functional as F
import numpy as np
import timeit
import unittest
from leaf import Tensor
from functools import partial

def _test_op(shapes, torch_func, leaf_func, name, timeits=10):
    torch_t = [torch.tensor(np.random.random(size=shape), requires_grad=True)
            for shape in shapes]
    leaf_t = [Tensor(t.detach().numpy(), requires_grad=True) for t in torch_t]

    torch_out = torch_func(*torch_t)
    leaf_out = leaf_func(*leaf_t)

    _arrout_t = isinstance(torch_out, tuple) or isinstance(torch_out, list)
    _arrout_l = isinstance(leaf_out, tuple) or isinstance(leaf_out, list)

    if isinstance(torch_out, torch.Tensor):
        torch_out = [torch_out]
    if isinstance(leaf_out, Tensor):
        leaf_out = [leaf_out]

    for tt, lt in zip(torch_out, leaf_out):
        np.testing.assert_allclose(tt.detach().numpy(), lt.data, atol=1e-6, rtol=1e-3)
        tt.mean().backward()
        lt.mean().backward()

    for tt, lt in zip(torch_t, leaf_t):
        np.testing.assert_allclose(tt.grad.detach().numpy(),
                lt.grad, atol=1e-6, rtol=1e-3)

    f_torch_ms = timeit.Timer(partial(torch_func, 
        *torch_t)).timeit(timeits) * 1000.0 / timeits
    f_leaf_ms = timeit.Timer(partial(leaf_func,
        *leaf_t)).timeit(timeits) * 1000.0 / timeits
    
    b_torch_ms = timeit.Timer(partial(lambda f,t,b: f(*t)[0].mean().backward() if b else
        f(*t).mean().backward(), torch_func, torch_t, _arrout_t)).timeit(timeits) * 1000.0 / timeits
    b_leaf_ms = timeit.Timer(partial(lambda f,t,b: f(*t)[0].mean().backward() if b else
        f(*t).mean().backward(), leaf_func, leaf_t, _arrout_l)).timeit(timeits) * 1000.0 / timeits

    print(f'\n[*] testing {name} with shapes {shapes}, torch/leaf \n' \
            f'forward: {f_torch_ms:.3f} ms / {f_leaf_ms:.3f} ms ' \
            f'backward: {b_torch_ms:.3f} ms / {b_leaf_ms:.3f} ms')
    
class TestOps(unittest.TestCase):
    def test_add(self):
        _test_op([(100, 100), (100, 100)], lambda x, y: x + y, Tensor.add, 'add')

    def test_sub(self):
        _test_op([(100, 100), (100, 100)], lambda x, y: x - y, Tensor.sub, 'sub')

    def test_matmul(self):
        _test_op([(128, 784), (784, 64)], lambda x, y: x.matmul(y), Tensor.matmul, 'matmul')

    def test_mean(self):
        _test_op([(78, 152)], lambda t: t.mean(), Tensor.mean, 'mean')
        _test_op([(20, 15, 2)], lambda t: t.mean(dim=1, keepdims=True),
                lambda t: Tensor.mean(t, axis=1, keepdims=True), 'mean-args')
        _test_op([(100, 15, 32, 72)], lambda t: t.mean(dim=(2, 3), keepdims=True),
                lambda t: Tensor.mean(t, axis=(2, 3), keepdims=True), 'mean-args')
        _test_op([(30, 40, 50)], lambda t: t.mean(dim=(0, 1, 2), keepdims=False),
                lambda t: Tensor.mean(t, axis=(0, 1, 2), keepdims=False), 'mean-args')

    def test_sum(self):
        _test_op([(40, 784)], lambda t: t.sum(), Tensor.sum, 'sum')
        _test_op([(64, 1, 1, 8)], lambda t: t.sum(axis=1, keepdims=True),
                lambda t: Tensor.sum(t, axis=1, keepdims=True), 'sum-args')
        _test_op([(14, 51, 7)], lambda t: t.sum(axis=(1, 2), keepdims=True),
                lambda t: Tensor.sum(t, axis=(1, 2), keepdims=True), 'sum-args')
        _test_op([(5, 10, 8, 1)], lambda t: t.sum(axis=1, keepdims=True),
                lambda t: Tensor.sum(t, axis=1, keepdims=True), 'sum-args')
        _test_op([(2, 4, 1)], lambda t: t.sum(axis=(0, 1), keepdims=False),
                lambda t: Tensor.sum(t, axis=(0, 1), keepdims=False), 'sum-args')

    def test_exp(self):
        _test_op([(120, 156, 80)], lambda t: t.exp(), Tensor.exp, 'exp')
        _test_op([(8, )], lambda t: t.exp(), Tensor.exp, 'exp')

    def test_log(self):
        _test_op([(154, 78, 2, 201)], lambda t: t.log(), Tensor.log, 'log')
        _test_op([(28, )], lambda t: t.log(), Tensor.log, 'log')
    
    def test_sigmoid(self):
        _test_op([(123, 51, 2)], lambda t: t.sigmoid(), Tensor.sigmoid, 'sigmoid')
        _test_op([(5, 6)], lambda t: t.sigmoid(), Tensor.sigmoid, 'sigmoid')

    def test_pow(self):
        _test_op([(1, 4)], lambda t: t.pow(2), lambda t: Tensor.pow(t, 2), 'pow')
        _test_op([(10, 2, 8)], lambda t: t.pow(0.5), lambda t: Tensor.pow(t, 0.5), 'pow')
        _test_op([(3, )], lambda t: t.pow(torch.tensor([1.0, 2.0, 0.5])),
                lambda t: Tensor.pow(t, (1.0, 2.0, 0.5)), 'pow-axis')
    
    def test_reshape(self):
        _test_op([(8, 2)], lambda t: torch.reshape(t, (4, 4)), 
                lambda t: Tensor.reshape(t, (4, 4)), 'reshape')
        _test_op([(17, 4, 8, 1)], lambda t: torch.reshape(t, (17, 2, 16)),
                lambda t: Tensor.reshape(t, (17, 2, 16)), 'reshape')

    def test_relu(self):
        _test_op([(64, 1, 28, 28)], lambda t: F.relu(t), Tensor.relu, 'relu')
        _test_op([(200, )], lambda t: F.relu(t), Tensor.relu, 'relu')

    def test_tanh(self):
        _test_op([(64, 1, 28, 28)], lambda t: torch.tanh(t), Tensor.tanh, 'tanh')
        _test_op([(10, 64, 30)], lambda t: torch.tanh(t), Tensor.tanh, 'tanh')
    
    def test_chunk(self):
        _test_op([(1, 12)], lambda t: t.chunk(4, dim=1), lambda t: Tensor.chunk(t,
            chunks=4, dim=1), 'chunk')
    
    def test_multiply(self):
        _test_op([(4, 8), (4, 8)], lambda a, b: torch.mul(a, b), lambda a, b:
                Tensor.multiply(a, b), 'multiply')
        _test_op([(128, 10, 49), (128, 10, 49)], lambda a, b: torch.mul(a, b),
                lambda a, b: Tensor.multiply(a, b), 'multiply')

