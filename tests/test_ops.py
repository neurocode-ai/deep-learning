import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timeit
import unittest
import leaf
from leaf import Tensor
from functools import partial
np.random.seed(1)

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
    def test_add1(self):
        _test_op([(100, 100), (100, 100)], lambda x, y: x + y, Tensor.add, 'add')
    def test_add2(self):
        _test_op([(100, 100, 40, 2), (100, 100, 40, 2)], lambda x, y: x + y, Tensor.add, 'add')

    def test_sub1(self):
        _test_op([(100, 100), (100, 100)], lambda x, y: x - y, Tensor.sub, 'sub')
    def test_sub2(self):
        _test_op([(100, 10, 30, 40), (100, 10, 30, 40)], lambda x, y: x - y, Tensor.sub, 'sub')

    def test_matmul1(self):
        _test_op([(128, 784), (784, 64)], lambda x, y: x.matmul(y), Tensor.matmul, 'matmul')
    def test_matmul2(self):
        _test_op([(2048, 2048), (2048, 2048)], lambda x, y: x.matmul(y), Tensor.matmul, 'matmul')

    def test_mean1(self):
        _test_op([(78, 152)], lambda t: t.mean(), Tensor.mean, 'mean')
    def test_mean2(self):
        _test_op([(20, 15, 2)], lambda t: t.mean(dim=1, keepdims=True),
                lambda t: Tensor.mean(t, axis=1, keepdims=True), 'mean-args')
    def test_mean3(self):
        _test_op([(100, 15, 32, 72)], lambda t: t.mean(dim=(2, 3), keepdims=True),
                lambda t: Tensor.mean(t, axis=(2, 3), keepdims=True), 'mean-args')
    def test_mean4(self):
        _test_op([(30, 40, 50)], lambda t: t.mean(dim=(0, 1, 2), keepdims=False),
                lambda t: Tensor.mean(t, axis=(0, 1, 2), keepdims=False), 'mean-args')

    def test_sum1(self):
        _test_op([(40, 784)], lambda t: t.sum(), Tensor.sum, 'sum')
    def test_sum2(self):
        _test_op([(64, 1, 1, 8)], lambda t: t.sum(axis=1, keepdims=True),
                lambda t: Tensor.sum(t, axis=1, keepdims=True), 'sum-args')
    def test_sum3(self):
        _test_op([(14, 51, 7)], lambda t: t.sum(axis=(1, 2), keepdims=True),
                lambda t: Tensor.sum(t, axis=(1, 2), keepdims=True), 'sum-args')
    def test_sum4(self):
        _test_op([(5, 10, 8, 1)], lambda t: t.sum(axis=1, keepdims=True),
                lambda t: Tensor.sum(t, axis=1, keepdims=True), 'sum-args')
    def test_sum5(self):
        _test_op([(2, 4, 1)], lambda t: t.sum(axis=(0, 1), keepdims=False),
                lambda t: Tensor.sum(t, axis=(0, 1), keepdims=False), 'sum-args')

    def test_exp1(self):
        _test_op([(120, 56, 80)], lambda t: t.exp(), Tensor.exp, 'exp')
    def test_exp2(self):
        _test_op([(8, )], lambda t: t.exp(), Tensor.exp, 'exp')

    def test_log1(self):
        _test_op([(154, 78, 2, 80)], lambda t: t.log(), Tensor.log, 'log')
    def test_log2(self):
        _test_op([(28, )], lambda t: t.log(), Tensor.log, 'log')
    
    def test_sigmoid1(self):
        _test_op([(123, 51, 2)], lambda t: t.sigmoid(), Tensor.sigmoid, 'sigmoid')
    def test_sigmoid2(self):
        _test_op([(5, 6)], lambda t: t.sigmoid(), Tensor.sigmoid, 'sigmoid')

    def test_pow1(self):
        _test_op([(1, 4)], lambda t: t.pow(2), lambda t: Tensor.pow(t, Tensor(2)), 'pow')
    def test_pow2(self):
        _test_op([(10, 2, 8)], lambda t: t.pow(0.5), lambda t: Tensor.pow(t, Tensor(0.5)), 'pow')
    def test_pow3(self):
        _test_op([(3, )], lambda t: t.pow(torch.tensor([1.0, 2.0, 0.5])),
                lambda t: Tensor.pow(t, Tensor((1.0, 2.0, 0.5))), 'pow-axis')
    
    def test_reshape1(self):
        _test_op([(8, 2)], lambda t: torch.reshape(t, (4, 4)), 
                lambda t: Tensor.reshape(t, (4, 4)), 'reshape')
    def test_reshape2(self):
        _test_op([(17, 4, 8, 1)], lambda t: torch.reshape(t, (17, 2, 16)),
                lambda t: Tensor.reshape(t, (17, 2, 16)), 'reshape')

    def test_relu1(self):
        _test_op([(64, 1, 28, 28)], lambda t: F.relu(t), Tensor.relu, 'relu')
    def test_relu2(self):
        _test_op([(200, )], lambda t: F.relu(t), Tensor.relu, 'relu')

    def test_tanh1(self):
        _test_op([(64, 1, 28, 28)], lambda t: torch.tanh(t), Tensor.tanh, 'tanh')
    def test_tanh2(self):
        _test_op([(10, 64, 30)], lambda t: torch.tanh(t), Tensor.tanh, 'tanh')
    
    def test_chunk1(self):
        _test_op([(1, 12)], lambda t: t.chunk(4, dim=1), lambda t: Tensor.chunk(t,
            chunks=4, dim=1), 'chunk')
    def test_chunk2(self):
        _test_op([(8, 4, 10)], lambda t: t.chunk(2, dim=0), lambda t: Tensor.chunk(t,
            chunks=2, dim=0), 'chunk')
    def test_chunk3(self):
        _test_op([(100, 10, 20, 50)], lambda t: t.chunk(10, dim=3), lambda t:
                Tensor.chunk(t, chunks=10, dim=3), 'chunk')
    
    def test_multiply1(self):
        _test_op([(4, 8), (4, 8)], lambda a, b: torch.mul(a, b), lambda a, b:
                Tensor.multiply(a, b), 'multiply')
    def test_multiply2(self):
        _test_op([(128, 10, 49), (128, 10, 49)], lambda a, b: torch.mul(a, b),
                lambda a, b: Tensor.multiply(a, b), 'multiply')

    def test_logsoftmax1(self):
        _test_op([(128, 100)], lambda x: nn.LogSoftmax(dim=1)(x), Tensor.logsoftmax,
        'logsoftmax')
    def test_logsoftmax2(self):
        _test_op([(512, 1)], lambda x: nn.LogSoftmax(dim=1)(x), Tensor.logsoftmax,
        'logsoftmax')

    def test_slice1(self):
        _test_op([(128, 4, 64)], lambda t: t[:, 0, :], lambda t: t[:, 0, :], 'slice')
    def test_slice2(self):
        _test_op([(64, 200)], lambda t: t[:2, :], lambda t: t[:2, :], 'slice')
    def test_slice3(self):
        _test_op([(100, 10, 20)], lambda t: t[:, :, :], lambda t: t[:, :, :], 'slice')
    def test_slice4(self):
        _test_op([(7, 5, 4, 8)], lambda t: t[0, 1, 2, 3], lambda t: t[0, 1, 2, 3], 'slice')

    def test_cat1(self):
        _test_op([(5, 3, 4) for _ in range(3)], 
                lambda x, y, z: torch.cat((x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)), dim=0),
                lambda x, y, z: leaf.concatenate((x, y, z), dim=0), 'concatenate-dim=0')
    def test_cat2(self):
        _test_op([(128, 5, 10) for _ in range(3)], 
                lambda x, y, z: torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1),
                lambda x, y, z: leaf.concatenate((x, y, z), dim=1), 'conatenate-dim=1')

    def test_conv2d1(self):
        _test_op([(1, 1, 28, 28), (4, 1, 3, 3)], lambda x, w: torch.nn.functional.conv2d(x, w, stride=1, groups=1),
                Tensor.conv2d, 'conv2d')

