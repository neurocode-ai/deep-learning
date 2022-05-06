import sys
import os
import numpy as np
import unittest
from edugrad import Tensor

class TestTensor(unittest.TestCase):
    def test_shape(self):
        t1 = Tensor([3, 2, 4])
        t2 = Tensor([[3], [2], [4]])
        t3 = Tensor(2)
        t4 = Tensor(-4.1)
        t5 = Tensor((5, 1, 3))
        t6 = Tensor(np.ones((5, 1, 4, 2)))
        
        assert t1.shape == (3, )
        assert t2.shape == (3, 1)
        assert t3.shape == (1, )
        assert t4.shape == (1, )
        assert t5.shape == (3, )
        assert t6.shape == (5, 1, 4, 2)

    def test_dtype(self):
        t1 = Tensor([-4, 1, 190])
        t2 = Tensor([[5, -2, 20]], dtype=np.uint8)
        t3 = Tensor(2.1)
        t4 = Tensor(-5, dtype=np.uint8)
        t5 = Tensor(np.zeros((5, 1, 2, 3)), dtype=np.uint8)
        t6 = Tensor(np.ones((8, 3, 4, 1)), dtype=np.int64)
        t7 = Tensor(np.random.uniform(-1, 1, size=(5, 1, 4)))

        assert t1.dtype == np.float32
        assert t2.dtype == np.uint8
        assert t3.dtype == np.float32
        assert t4.dtype == np.uint8
        assert t5.dtype == np.uint8
        assert t6.dtype == np.int64
        assert t7.dtype == np.float32

    def test_classmethods(self):
        t1 = Tensor.zeros(4, 2, 3, dtype=np.uint8)
        t2 = Tensor.ones(8, 5, 2, 4, requires_grad=True)
        t3 = Tensor.diagonal(4, dtype=np.float64)
        t4 = Tensor.uniform(5, 12, 3, low=-2.0, high=-1.0)
        t5 = Tensor.normal(4, loc=1.0, scale=0.5, dtype=np.float64)
        t6 = Tensor.full(2, 7, 5, 1, 5, 6, requires_grad=True)

        assert t1.shape == (4, 2, 3)
        assert t1.dtype == np.uint8
        assert t2.shape == (8, 5, 2, 4)
        assert t2.dtype == np.float32
        assert t2.requires_grad
        assert t3.shape == (4, 4)
        assert t3.dtype == np.float64
        assert not t3.requires_grad
        assert t4.shape == (5, 12, 3)
        assert t4.dtype == np.float32
        assert t5.shape == (4, )
        assert t5.dtype == np.float64
        assert t6.shape == (7, 5, 1, 5, 6)
        assert t6.requires_grad

        np.testing.assert_array_less(t4.data, np.ones((5, 12, 3)))
        np.testing.assert_array_equal(t6.data, np.full((7, 5, 1, 5, 6), 2))



