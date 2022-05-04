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

