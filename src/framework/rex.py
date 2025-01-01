import torch
import numpy as np
from src.framework.graph import Graph
#from src.framework.rexbase import RexBase as RexBase
from src.framework.rexbase import RexDictBase as RexBase

class RexIdentity(RexBase):
  def __init__(self):
    super().__init__()
    self.class_name = "TRex.Identity"

class RexReshape(RexBase):
  def __init__(self, args={}):
    super().__init__()
    self.class_name = "TRex.Reshape"
    self.arguments  = {
      "target_shape": (-1,)
    }
    self.update_args(args)

class RexZeroPadding2D(RexBase):
  def __init__(self, args={}):
    super().__init__()
    self.class_name = "TRex.ZeroPadding2D"
    self.arguments  = {
      "padding" : 0
    }
    self.update_args(args)

class RexConv2D(RexBase):
  def __init__(self, args={}):
    super().__init__()
    self.class_name = "TRex.Conv2D"
    self.arguments  = {
      "in_channels"       : -1,
      "filters"           : -1,
      "kernel_size"       : 1,
      "strides"           : (1,1),
      "padding"           : "valid",
      "dilation_rate"     : (1,1),
      "use_bias"          : True,
      "activation"        : None
    }
    self.update_args(args)
    self.arguments["padding"] = "'valid'" if self.arguments["padding"] == "valid" else "'same'"


class RexDepthwiseConv2D(RexBase):
  def __init__(self, args={}):
    super().__init__()
    self.class_name = "TRex.DepthwiseConv2D"
    self.arguments  = {
      "in_channels"       : -1,
      "kernel_size"       : -1,
      "strides"           : (1,1),
      "dilation_rate"     : (1,1),
      "padding"           : "valid",
      "use_bias"          : True,
      "activation"        : None
    }
    self.update_args(args)
    self.arguments["padding"] = "'valid'" if self.arguments["padding"] == "valid" else "'same'"

class RexGlobalAveragePooling2D(RexBase):
  def __init__(self, args={}):
    super().__init__()
    self.class_name = "TRex.GlobalAveragePooling2D"
    self.arguments  = {
      "kernel_size"       : (1,1),
    }
    self.update_args(args)


class RexModule(RexBase, Graph):
  def __init__(self, class_name="MyModule", object_name="module", use_module_dict=False):
    super(RexModule, self).__init__(standalone=True, class_name=class_name, object_name=object_name)
    self.graph      = {}
    self.state_dict = {}
    self.use_module_dict = use_module_dict

  def add_input_layer(self, y_id, input_shape=(416,416,3)):
    self.input_shape = input_shape
    self.add_layer(None, y_id, [], weights=None)

  def get_name(self, x):
    if isinstance(x, (int, str)):
      return f"_{x}"
    return x.name

  def add_layer(self, data : RexBase, y_id, x_ids, weights = None):
    vertex_name = self.get_name(y_id)
    parent_vertexes = [self.get_name(_) for _ in x_ids] if len(x_ids)>0 else None
    if data:
      data.set_object_name(vertex_name)
    self.add_vertex(vertex_name, parent_vertexes=parent_vertexes, data=data)

    # k.shape = (H,W,Cin,Cout) -> (Cout,Cin,H,W)
    if weights != None:
      if isinstance(data, RexDepthwiseConv2D):
        k = np.moveaxis(weights[0], [2,3], [0,1])
      else:
        k = np.moveaxis(weights[0], [2,3], [1,0])
      prefix = "d." if self.use_module_dict else ""
      self.state_dict[prefix+vertex_name+".conv.weight"] = torch.nn.Parameter(torch.Tensor(k), requires_grad=True)
      if len(weights) > 1:
        self.state_dict[prefix+vertex_name+".conv.bias"] = torch.nn.Parameter(torch.Tensor(weights[1]), requires_grad=True)

  def set_output(self, outputs):
    self.outputs = [self.get_name(_) for _ in outputs]

  def gen_input(self, name):
    text = f"{name} = torch.randn(1,{self.input_shape[2]},{self.input_shape[0]},{self.input_shape[1]})"
    return text

  def save_state_dict(self, filename=""):
    filename = self.class_name+".pt" if filename == "" else filename
    torch.save(self.state_dict, filename)
  
  def gen_load_weight(self, filename=""):
    filename = self.class_name+".pt" if filename == "" else filename
    return f"{self.object_name}.load_state_dict(torch.load('{filename}'))"

  def gen_code(self):
    text = f"class {self.class_name}(torch.nn.Module):\n  def __init__(self):\n    super().__init__()\n"
    if self.use_module_dict:
      text += "    self.d = torch.nn.ModuleDict({})\n"
    text += f"    self.input_shape = (1,{self.input_shape[0]},{self.input_shape[1]},{self.input_shape[2]})\n"
    for k in self.graph:
      if self.graph[k]['data']:
        text += "    " + self.graph[k]['data'].gen_init() + "\n"
    text += f"\n  def forward(self, t1):\n"

    var, output = 0, {}
    for k in self.graph:
      if self.graph[k]['data']:
        arg = []
        for p in self.graph[k]['parents']:
          arg.append(output[p])
        text += "    " + self.graph[k]['data'].gen_call(f"t{var+1}", arg) + "\n"

      output[k] = f"t{var+1}"
      var += 1

    ret   = ','.join([output[_] for _ in self.outputs])
    text += f"    return {ret}\n"

    return text