from torch.autograd import Function
from build import mathutils


class BroadcastAccum(Function):
  """Accumulate x += y using broadcasting sum.

  By default, pytorch does not support broadcasting sum. Instead, it expand the
  tensor to do the broadcasting sum, which is inefficient use of memory.
  """
  def forward(self, x, y):
    return mathutils.broadcast_sum(x, y)
