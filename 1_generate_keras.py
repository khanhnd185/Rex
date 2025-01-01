import keras
import argparse
from src.framework.optimizer   import optimize
from vision.imagenetclassifier import ImagenetVerifierFactory

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Generate Keras model')
  parser.add_argument('--datadir', default="./data", type=str, help="dataset path")
  args = parser.parse_args()

  preproc_fn   = lambda x:x/127.5-1
  train_model  = keras.applications.MobileNet()
  fuse_preproc = False
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

  optimized_model.save("output/mobilenetv1_opt.keras")