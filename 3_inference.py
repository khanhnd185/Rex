import torch
from output.mobilenetv1 import Mobilenetv1
from output.mobilenetv2 import Mobilenetv2

if __name__=="__main__":
  for model_name, model in [
    ["mobilenetv1", Mobilenetv1()],
    ["mobilenetv2", Mobilenetv2()],
  ]:
    x = torch.moveaxis(torch.Tensor(*model.input_shape), -1, 1) # NHWC -> NCHW
    model.load_state_dict(torch.load(f'output/{model_name}.pt'))
    y = model(x)
    print(y.shape)

    # Access SRT layer attribute
    for k,v in model.d.items():
      print(f"{k}: {v.task}")