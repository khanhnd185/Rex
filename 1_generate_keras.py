import keras
import argparse
from src.framework.optimizer   import optimize
from src.vision.imagenetclassifier import ImagenetVerifierFactory

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Generate Keras model')
  parser.add_argument('--datadir', default="./data", type=str, help="dataset path")
  args = parser.parse_args()

  preproc_fn   = lambda x:x/127.5-1
  fuse_preproc = False
  
  for train_model, model_name in [
    ["mobilenetv1", keras.applications.MobileNet  ()],
    ["mobilenetv2", keras.applications.MobileNetV2()],
  ]:
    train_model.summary()
    verifier = ImagenetVerifierFactory(args.datadir)
    verifier.evaluate(
      train_model
      , preproc_fn=preproc_fn
      , images=1000
      , batch_size=32
      , display=True
      , shuffle_seed=123
    )

    optimized_model = optimize(train_model, fuse_preproc=fuse_preproc, preproc_fn=preproc_fn)
    optimized_model.summary()

    if fuse_preproc:
        verifier.evaluate(
          optimized_model
          , preproc_fn=lambda x:x/256.0
          , images=1000
          , batch_size=32
          , display=True
          , shuffle_seed=123
        )
    else:
        verifier.evaluate(
          optimized_model
          , preproc_fn=preproc_fn
          , images=1000
          , batch_size=32
          , display=True
          , shuffle_seed=123
        )
    optimized_model.save(f"output/{model_name}_opt.keras")
