from keras import Model
from keras.layers import (
  ReLU,
  InputLayer,
  Conv2D,
  DepthwiseConv2D,
  Reshape,
  GlobalAveragePooling2D,
  ZeroPadding2D,
)

from src.framework.rex import (
  RexModule,
  RexIdentity,
  RexConv2D,
  RexDepthwiseConv2D,
  RexZeroPadding2D,
  RexReshape,
  RexGlobalAveragePooling2D,
)

class Keras2RexGenerator():
  def __init__(
        self
      , model
      , model_name
      , use_module_dict=False
    ):
    assert isinstance(model, Model), f"model='{model}' must be a 'keras.Model' instance"

    self.use_module_dict = use_module_dict
    self.model              = model
    self.model_name         = model_name
    self.quant_mode         = 0
    self.quant_bits_signal  = 0

  def create_trex_model(
        self
      , output_dir="output"
      , source_file=""
      , weight_file=""
    ):
    print(f"[I] Create Trex Torch model with model '{self.model.name}'.")
    output_layers = [_.node.outbound_layer for _ in self.model.outputs]

    module = RexModule(
                class_name=self.model_name.capitalize()
                , object_name="module"
                , use_module_dict=self.use_module_dict
              )
    module.set_output(output_layers)

    for L0 in self.model.layers:
      LN1 = L0.inbound_nodes[0].inbound_layers

      if   (hasattr   (L0, "removed")):
        module.add_layer(RexIdentity(), L0, LN1)
      elif (isinstance(L0, InputLayer)):
        module.add_input_layer(L0, input_shape=L0.input_shape[0][1:])
      elif (isinstance(L0, (Conv2D, DepthwiseConv2D))):
        weights = [_.numpy() for _ in L0.weights]
        args  = {
          "in_channels"       : L0.input_shape[-1], # Torch exclusive argument
          "kernel_size"       : L0.kernel_size,
          "strides"           : L0.strides,
          "padding"           : L0.padding,
          "use_bias"          : L0.use_bias,
          "dilation_rate"     : L0.dilation_rate,
          "activation"        : "'relu6'" if isinstance(L0.activation, ReLU) else None,
        }
        if (isinstance(L0, Conv2D)):
              args["filters"] = L0.filters
              _layer = RexConv2D(args=args)
        else: _layer = RexDepthwiseConv2D(args=args)
        module.add_layer(_layer, L0, [LN1], weights=weights)
      elif (isinstance(L0, GlobalAveragePooling2D)):
        args  = {
          "kernel_size"       : L0.input_shape[1:3], # Torch exclusive argument
        }
        module.add_layer(RexGlobalAveragePooling2D(args=args), L0, [LN1])
      elif (isinstance(L0, Reshape)):
        #args = {"shape":(-1, *L0.output_shape[1:])}
        module.add_layer(RexReshape(args={'target_shape':(-1, *L0.output_shape[1:])}), L0, [LN1])
      elif (isinstance(L0, ZeroPadding2D)):
        module.add_layer(RexZeroPadding2D(args={'padding':L0.padding}), L0, [LN1])
      else: raise NotImplementedError(f"Unknown layer '{L0.__class__.__name__}' provided")

    print(module.summary())
    module.save_state_dict(filename=f"{output_dir}/{weight_file}")

    with open(f"{output_dir}/{source_file}", "w") as f:
      f.write("import torch\nimport src.rex_torch as TRex\n\n")
      f.write(module.gen_code())
      f.write("\n")
