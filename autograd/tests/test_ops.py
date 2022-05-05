import torch
import numpy as np
import timeit
import unittest
from autograd import Tensor
from functools import partial

def _test_op(shapes, torch_func, autograd_func, name, timeits=2):
    torch_t = [torch.tensor(np.random.random(size=shape), requires_grad=True)
            for shape in shapes]
    autograd_t = [Tensor(t.detach().numpy(), requires_grad=True) for t in torch_t]

    torch_out = torch_func(*torch_t)
    autograd_out = autograd_func(*autograd_t)

    np.testing.assert_allclose(torch_out.detach().numpy(), autograd_out.data, 
            atol=1e-6, rtol=1e-3)

    torch_out.mean().backward()
    autograd_out.mean().backward()        

    for tt, at in zip(torch_t, autograd_t):
        np.testing.assert_allclose(tt.grad.detach().numpy(),
                at.grad.data, atol=1e-6, rtol=1e-3)

    f_torch_ms = timeit.Timer(partial(torch_func, 
        *torch_t)).timeit(timeits) * 1000.0 / timeits
    f_autograd_ms = timeit.Timer(partial(autograd_func,
        *autograd_t)).timeit(timeits) * 1000.0 / timeits

    b_torch_ms = timeit.Timer(partial(lambda f,t: f(*t).mean().backward(),
        torch_func, torch_t)).timeit(timeits) * 1000.0 / timeits
    b_autograd_ms = timeit.Timer(partial(lambda f,t: f(*t).mean().backward(),
        autograd_func, autograd_t)).timeit(timeits) * 1000.0 / timeits

    print(f'\n[*] testing {name} with shapes {shapes}, torch/autograd \n' \
            f'forward: {f_torch_ms:.3f} ms / {f_autograd_ms:.3f} ms ' \
            f'backward: {b_torch_ms:.3f} ms / {b_autograd_ms:.3f} ms')
    
class TestOps(unittest.TestCase):
    def test_add(self):
        _test_op([(100, 100), (100, 100)], lambda x, y: x + y, Tensor.add, 'add')

    def test_sub(self):
        _test_op([(100, 100), (100, 100)], lambda x, y: x - y, Tensor.sub, 'sub')

    def test_mean(self):
        _test_op([(78, 152)], lambda t: t.mean(), Tensor.mean, 'mean')

