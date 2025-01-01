import tensorflow as tf
import keras
from keras.layers import *
from keras.src.layers.core.tf_op_layer import TFOpLambda
import numpy as np
import sys
import os
import tempfile


def build_model(
    inputs
  , outputs
  , name=None
):
  model       = keras.Model(inputs, outputs, name=name)
  model_file  = tempfile.mkstemp(suffix=".keras")[1]
  model.save(model_file)              # save-and-load the model for cleanup
  model       = keras.models.load_model(model_file)
  os.system(f"rm -rf {model_file}")
  return  model


def optimize(
    model
  , fuse_preproc=True
  , fuse_bgr2rgb=False
  , preproc_fn=None
):
  if (type(model)==str):
    model = keras.models.load_model(model)

  print(f"[I] Create the inference model '{model.name}'.")
  SIGNALS       = {}                  # record all signals to build the execution plan
  OUTPUT_LAYERS = [_.node.outbound_layer for _ in model.outputs]

  for layer_id, L0 in enumerate(model.layers):
    if (len(L0.inbound_nodes)>1): raise NotImplementedError
    LN1 = L0.inbound_nodes [0].inbound_layers
    L1  = L0.outbound_nodes[0].outbound_layer if (len(L0.outbound_nodes)>0) else None

    if   (hasattr(L0, "removed")):
      if (type(LN1)==list):
        if (isinstance(LN1[1], TFOpLambda) and (LN1[1].function.__name__ == 'multiply')):
          SIGNALS[L0] = SIGNALS[LN1[1]]
        else:
          SIGNALS[L0] = SIGNALS[LN1[0]]
      else:
        SIGNALS[L0] = SIGNALS[LN1]

    elif (isinstance(L0, InputLayer)):  # [InputLayer]
      INPUTS      = Input(shape=model.input_shape[1:], dtype=L0.dtype, name="images")
      SIGNALS[L0] = INPUTS
    elif (isinstance(L0, (Conv2D, DepthwiseConv2D))): # [Conv2D], [DepthwiseConv2D]
      use_bias    = L0.use_bias
      activation  = L0.activation

      if (isinstance(L1, BatchNormalization)):
        kernel      = L0.weights[0].numpy()
        bias        = L0.weights[1].numpy() if (use_bias)  else 0
        gamma       = L1.gamma.numpy()      if (L1.scale)  else 1
        beta        = L1.beta.numpy()       if (L1.center) else 0
        mean        = L1.moving_mean.numpy()
        std         = np.sqrt(L1.moving_variance+L1.epsilon)
        L1.removed  = True
        use_bias    = True

        if (isinstance(L0, Conv2D)):  k =                gamma/std*           kernel
        else:                         k = np.expand_dims(gamma/std*np.squeeze(kernel),-1)
        b  = beta + (gamma/std)*(bias-mean)
        LA = L1.outbound_nodes[0].outbound_layer if (len(L1.outbound_nodes)>0) else None

      else:                                     # Conv2D
        k  = L0.weights[0].numpy()
        b  = L0.weights[1].numpy() if (use_bias) else 0
        LA = L1

      if (fuse_preproc):
        f               = np.array([[0,0,0],[1,1,1]], dtype="float32")
        preproc_layers  = []

        while (True):
          if   (isinstance(LN1, InputLayer)): break
          elif (isinstance(LN1, (Rescaling, Normalization))):  preproc_layers.insert(0,LN1)
          LN1 = LN1.inbound_nodes[0].inbound_layers

        if   (len(preproc_layers)>0):
          for layer in preproc_layers:  f = layer(f)
          f   = np.squeeze(f)
        elif (preproc_fn!=None):
          f   = preproc_fn(f)
        else: raise ValueError(f"Invalid preproc_fn={preproc_fn} provided")

        b_shiftscale  = -f[0]
        k_scale       = (f[1]-f[0])*256
        k             = np.moveaxis(k, -1,  0)  # HWCN-to-NWHC
        b            -= np.sum(b_shiftscale*np.sum(k,axis=(1,2)), 1)
        k            *= k_scale
        if (fuse_bgr2rgb):  k[...,:] = k[...,::-1]
        k             = np.moveaxis(k,  0, -1)  # NHWC-to-HWCN
        use_bias      = True
        fuse_preproc  = False

      if   (L0.activation.__name__=="linear"):
        if (len(L0.outbound_nodes)==1 and isinstance(LA, (Activation, ReLU))):
          activation  = LA.activation if (isinstance(LA, Activation)) else LA
          LA.removed  = True

      kwargs  = {
        "kernel_size" : L0.kernel_size,
        "strides"     : L0.strides,
        "padding"     : L0.padding,
        "use_bias"    : use_bias,
        "activation"  : activation,
        "weights"     : [k,b] if (use_bias) else [k],
        "name"        : L0.name
      }
      if (isinstance(L0, Conv2D)):  _layer = Conv2D(L0.filters, **kwargs)
      else:                         _layer = DepthwiseConv2D(   **kwargs)

      SIGNALS[L0] = _layer(SIGNALS[LN1])
    elif (isinstance(L0, Dense)):
      if ((L0 in OUTPUT_LAYERS) and (L0.activation.__name__=="softmax")):
        L0.activation = keras.activations.linear

      SIGNALS[L0] = L0(SIGNALS[LN1])
    elif (isinstance(L0, Activation)):
      if ((L0 in OUTPUT_LAYERS) and (L0.activation.__name__=="softmax")):
        SIGNALS[L0] = SIGNALS[LN1]
      else:
        SIGNALS[L0] = L0(SIGNALS[LN1])
    elif (isinstance(L0, (Rescaling, Normalization))):
      if (fuse_preproc):  SIGNALS[L0] = SIGNALS[LN1]
      else:               SIGNALS[L0] = L0(SIGNALS[LN1])

    elif (isinstance(L0, Dropout)):   # [layers to remove]
      SIGNALS[L0] = SIGNALS[LN1]
    elif (isinstance(L0, TFOpLambda)):
      if ((L0.function.__name__ == '_add_dispatch') and
          (isinstance(L1, ReLU)) and
          (len(L1.outbound_nodes)>0) and
          (isinstance(L1.outbound_nodes[0].outbound_layer, TFOpLambda)) and
          (L1.outbound_nodes[0].outbound_layer.function.__name__ == 'multiply')):
        L2  = L1.outbound_nodes[0].outbound_layer
        L3  = L2.outbound_nodes[0].outbound_layer if (len(L2.outbound_nodes)>0) else None
        if ((isinstance(L3, Multiply)) and
            ((L3.inbound_nodes[0].inbound_layers[0] == LN1) or
            (L3.inbound_nodes[0].inbound_layers[1] == LN1))):
          SIGNALS[L0] = Activation('hard_swish')(SIGNALS[LN1])
          print(f"[{layer_id}] {L0.function.__name__ }: hard_swish")
          L1.removed  = True
          L2.removed  = True
          L3.removed  = True
        else:
          SIGNALS[L0] = Activation('hard_sigmoid')(SIGNALS[LN1])
          print(f"[{layer_id}] {L0.function.__name__ }: hardsigmoid")
          L1.removed  = True
          L2.removed  = True
      else:
        SIGNALS[L0] = L0([SIGNALS[_] for _ in LN1] if (type(LN1)==list) else SIGNALS[LN1])
    else:
      SIGNALS[L0] = L0([SIGNALS[_] for _ in LN1] if (type(LN1)==list) else SIGNALS[LN1])

  return build_model(INPUTS, [SIGNALS[_] for _ in OUTPUT_LAYERS], name=model.name)
