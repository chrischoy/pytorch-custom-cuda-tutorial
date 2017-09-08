from torch.autograd import Function
import mathutils


class BroadcastAccum(Function):
  """Accumulate x += y using broadcasting sum.
  """
  def forward(self, x, y):
    return mathutils.broadcast_sum(x, y)
