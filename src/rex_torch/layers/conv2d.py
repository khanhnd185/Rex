import torch

class Conv2D(torch.nn.Module):

  def __init__(self,
    in_channels,
    filters,
    kernel_size,
    strides=(1,1),
    padding="same",
    use_bias=True,
    activation=None,
    dilation_rate=(1,1),
    stride_offset=1,
    **kwargs
  ):
    super().__init__()

    if padding == 'same' and strides in [2, (2,2)]:
      padding = 'valid'
      self.zeropad = torch.nn.ZeroPad2d(padding=[0, 1, 0, 1])
    else:
      self.zeropad = torch.nn.Identity()

    self.conv       = torch.nn.Conv2d(
      in_channels,
      filters,
      kernel_size,
      stride=strides,
      padding=padding,
      dilation=dilation_rate,
      bias=use_bias,
    )
    if   activation == 'relu' : self.activation = torch.nn.ReLU()
    elif activation == 'relu6': self.activation = torch.nn.ReLU6()
    else: self.activation = torch.nn.Identity()

    ##  Initialize the layer-specific arguments.
    self.stride_offset      = stride_offset

  def forward(self, x):
    t0    = self.zeropad(x)
    t1    = self.conv(t0)
    y     = self.activation(t1)
    return  y


class DepthwiseConv2D(torch.nn.Module):

  def __init__(self,
    in_channels,
    kernel_size,
    strides=(1,1),
    padding="same",
    use_bias=True,
    activation=None,
    dilation_rate=(1,1),
    stride_offset=1,
    **kwargs
  ):
    super().__init__()

    ##  Initialize the layer structure.
    if padding == 'same' and strides in [2, (2,2)]:
      padding = 'valid'
      self.zeropad = torch.nn.ZeroPad2d(padding=[0, 1, 0, 1]) # TODO: Fix hardcoding padding
    else:
      self.zeropad = torch.nn.Identity()

    self.conv       = torch.nn.Conv2d(
      in_channels,
      in_channels,
      kernel_size,
      stride=strides,
      padding=padding,
      dilation=dilation_rate,
      bias=use_bias,
      groups=in_channels
    )
    if   activation == 'relu' : self.activation = torch.nn.ReLU()
    elif activation == 'relu6': self.activation = torch.nn.ReLU6()
    else: self.activation = torch.nn.Identity()

    ##  Initialize the layer-specific arguments.
    self.stride_offset      = stride_offset


  def forward(self, x):
    ##  Compute the output.
    t0    = self.zeropad(x)
    t1    = self.conv(t0)
    y     = self.activation(t1)

    return  y

