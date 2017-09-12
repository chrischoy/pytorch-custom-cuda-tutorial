import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable
import mathutils


class BroadcastAccum(Function):
  """Accumulate x += y using broadcasting sum.
  """
  def forward(self, x, y):
    mathutils.broadcast_sum(x, y, *map(int, x.size()))
    return x


class TestBroadcastAccum(unittest.TestCase):

  def test_broadcast_accum(self):
    N, M = 3, 5
    x = torch.rand(N, M).cuda()
    y = torch.rand(N, 1).cuda()

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    x_np += y_np

    x = BroadcastAccum()(Variable(x), Variable(y))
    self.assertTrue(np.allclose(x_np, x.data.cpu().numpy()))


if __name__ == '__main__':
  unittest.main()
