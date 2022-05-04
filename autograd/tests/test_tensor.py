import sys
import os
import numpy as np
import unittest
from autograd import Tensor

class TestTensor(unittest.TestCase):
    def test_shape(self):
        t1 = Tensor([3,2,4])
        t2 = Tensor([[3], [2], [4]])
        t3 = Tensor(2)
        t4 = Tensor(-4.1)
        t5 = Tensor((5,1,3))
        t6 = Tensor(np.ones((5,1,4,2)))
        
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
        t6 = Tensor(np.ones((8,3,4,1)), dtype=np.int64)
        t7 = Tensor(np.random.uniform(-1, 1, size=(5,1,4)))

        assert t1.dtype == np.float32
        assert t2.dtype == np.uint8
        assert t3.dtype == np.float32
        assert t4.dtype == np.uint8
        assert t5.dtype == np.uint8
        assert t6.dtype == np.int64
        assert t7.dtype == np.float32

