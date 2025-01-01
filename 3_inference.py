import torch
from output.mobilenetv1 import Mobilenetv1

if __name__=="__main__":
  m = Mobilenetv1()
  x = torch.moveaxis(torch.Tensor(*m.input_shape), -1, 1) # NHWC -> NCHW
  m.load_state_dict(torch.load('output/mobilenetv1.pt'))
  y = m(x)
  print(y.shape)

  # Access SRT layer attribute
  for k,v in m.d.items():
    print(f"{k}: {v.task}")