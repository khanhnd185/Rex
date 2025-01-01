import torch

class GlobalAveragePooling2D(torch.nn.Module):
  
  def __init__(self, 
    kernel_size,
    keepdims=True,
    **kwargs
  ):
    super().__init__()

    self.pool               = torch.nn.AvgPool2d(
      kernel_size=kernel_size
    )
    self.keepdims = keepdims

  def forward(self, x):
    y  = self.pool(x)
    if not self.keepdims:
      y = torch.reshape(y, y.shape[:-2])
    return  y
