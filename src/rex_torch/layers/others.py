import torch
import sys


class Reshape(torch.nn.Module):

  def __init__(self,
    target_shape,
    **kwargs
  ):
    super().__init__()
    self.task  = None
    self.shape = target_shape

  def forward(self, x):
    return torch.reshape(x, self.shape)



class ReLU(torch.nn.Module):

  def __init__(self,
    **kwargs
  ):
    super().__init__()
    self.task  = None
    self.relu  = torch.nn.ReLU()


  def forward(self, x):
    return  self.relu(x)



class ZeroPadding2D(torch.nn.Module):

  def __init__(self,
    padding,
    **kwargs
  ):
    super().__init__()
    padding    = (padding[1][0], padding[1][1], padding[0][0], padding[0][1])
    self.task  = None
    self.pad   = torch.nn.ZeroPad2d(padding=padding)


  def forward(self, x):
    return  self.pad(x)


  def export(self, tensors, hardware_config, fout=sys.stdout):
    fout.write(f"""\
  // {__class__.__name__} {self.name}

""")

##  End of class ZeroPadding2D


class Identity(torch.nn.Module):

  def __init__(self,
    **kwargs
  ):
    self.task  = None
    super().__init__()

  def forward(self, x):
    return  x

  def export(self, tensors, hardware_config, fout=sys.stdout):
    fout.write(f"""\
  // {__class__.__name__} {self.name}

""")

##  End of class Softmax
