import torch

class Dense(torch.nn.Module):

  def __init__(self,
    in_channels,
    units,
    activation=None,
    use_bias=True,
    **kwargs
  ):
    super().__init__()
    self.linear = torch.nn.Linear(in_channels, units, bias=use_bias)
    if   activation == 'relu' : self.activation = torch.nn.ReLU()
    elif activation == 'relu6': self.activation = torch.nn.ReLU6()
    else: self.activation = torch.nn.Identity()

  def forward(self, x):
    t = self.linear(x)
    y = self.activation(t)
    return y
  