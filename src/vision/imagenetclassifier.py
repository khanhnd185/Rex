import tensorflow as tf
import keras
import torch
import torchvision
import numpy as np
import abc
import tqdm


def ImagenetVerifierFactory(
    validate_dir
  , framework="keras"
  , validate_images=50000
):
  if   framework=="keras":  verifier_fn = ImageNetVerifierKeras
  elif framework=="tflite": verifier_fn = ImageNetVerifierTFLite
  elif framework=="torch":  verifier_fn = ImageNetVerifierTorch
  else: raise NotImplementedError(f"Invalid framework={framework} provided")
  return  verifier_fn(validate_dir, validate_images, framework)


class ImageNetVerifier:

  def __init__(self, validate_dir, validate_images, framework):
    self.validate_dir     = validate_dir
    self.validate_images  = validate_images
    self.framework        = framework


  @abc.abstractmethod
  def evaluate(self): raise NotImplementedError


  def _evaluate_accuracy(self):
    top1,top5 = 0,0
    n         = 0

    for inputs, labels in tqdm.tqdm(self._dataset):
      predicts  = self.model_runner(inputs)

      if   self.framework=="tflite":
        predicts  = predicts[self.output_name]
      elif self.framework=="torch":
        predicts  = predicts.detach().numpy()
        labels    = labels.numpy()

      predicts5 = predicts.argsort()[:,[-1,-2,-3,-4,-5]]
      top5     += np.sum([np.any(l==p5) for l, p5 in zip(labels, predicts5)])
      top1     += np.sum(labels==predicts5[:,0])
      n        += inputs.shape[0]
    return  top1/n, top5/n


class ImageNetVerifierKeras(ImageNetVerifier):

  def __init__(self, *args):
    super().__init__(*args)


  def evaluate(self,
    model, image_size=None, preproc_fn=None,
    images=None, batch_size=32, shuffle_seed=None, display=True
  ):
    if type(model)==str:
      print(f"[I] Load the Keras model '{model}'")
      model = keras.models.load_model(model)
    if preproc_fn==None:  preproc_fn = lambda x:x

    self.model        = model
    self.model_runner = lambda x:model.predict(preproc_fn(x), batch_size=batch_size, verbose=0)
    self.input_shape  = model.input_shape

    self._load_dataset(image_size, images, batch_size, shuffle_seed)
    acc1, acc5        = self._evaluate_accuracy()
    if display:
      print(f"[I] Measured accuracy for the Keras model '{model.name}'")
      print( "    Top1=%.4f, Top5=%.4f" %(acc1, acc5))
    return  acc1, acc5


  def _load_dataset(self, image_size, images, batch_size, shuffle_seed):
    dataset   = keras.utils.image_dataset_from_directory(
      self.validate_dir,
      label_mode="int",
      batch_size=None,
      image_size=image_size or self.model.input_shape[-3:-1],
      shuffle=True,
      seed=shuffle_seed,
      crop_to_aspect_ratio=True
    )
    if image_size:
      offsets = (np.array(image_size)-self.input_shape[-3:-1])//2
      dataset = dataset.map(lambda images, labels: (
        tf.image.crop_to_bounding_box(images, *offsets, *self.input_shape[-3:-1]), labels
      ), num_parallel_calls=tf.data.AUTOTUNE)

    if images:
      dataset = dataset.take(images)
      print(f"[I] Validate for {images} images from '{self.validate_dir}'")
    else:
      print(f"[I] Validate for {self.validate_images} images from '{self.validate_dir}'")
    self._dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class ImageNetVerifierTFLite(ImageNetVerifier):

  def __init__(self, *args):
    super().__init__(*args)


  def evaluate(self,
    model, image_size=None, preproc_fn=None,
    images=None, batch_size=32, shuffle_seed=None, display=True
  ):
    if type(model)==str:
      print(f"[I] Load the TFLite model '{model}'")
      model = tf.lite.Interpreter(model)
    if preproc_fn==None:  preproc_fn = lambda x:x

    model_runner      = model.get_signature_runner()
    input_detail      = model_runner.get_input_details().popitem()
    self.model        = model
    self.model_runner = lambda x:model_runner(**{input_detail[0]:preproc_fn(x)})
    self.input_shape  = input_detail[1]["shape"]
    self.output_name  = model_runner.get_output_details().popitem()[0]

    self._load_dataset(image_size, images, batch_size, shuffle_seed)
    acc1, acc5        = self._evaluate_accuracy()
    if display:
      print(f"[I] Measured accuracy for the TFLite model")
      print( "    Top1=%.4f, Top5=%.4f" %(acc1, acc5))
    return  acc1, acc5


  def _load_dataset(self, image_size, images, batch_size, shuffle_seed):
    dataset = keras.utils.image_dataset_from_directory(
      self.validate_dir,
      label_mode="int",
      batch_size=None,
      image_size=image_size or self.input_shape[-3:-1],
      shuffle=True,
      seed=shuffle_seed,
      crop_to_aspect_ratio=True
    )
    if image_size:
      offsets = (np.array(image_size)-self.input_shape[-3:-1])//2
      dataset = dataset.map(lambda images, labels: (
        tf.image.crop_to_bounding_box(images, *offsets, *self.input_shape[-3:-1]), labels
      ), num_parallel_calls=tf.data.AUTOTUNE)

    if images:
      dataset = dataset.take(images)
      print(f"[I] Validate for {images} images from '{self.validate_dir}'")
    else:
      print(f"[I] Validate for {self.validate_images} images from '{self.validate_dir}'")
    self._dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class ImageNetVerifierTorch(ImageNetVerifier):

  def __init__(self, *args):
    super().__init__(*args)


  def evaluate(self,
    model,
    transform=None, images=None, batch_size=32, shuffle_seed=None, display=True
  ):
    self.model        = model
    self.model_runner = model.eval()

    self._load_dataset(images, batch_size, shuffle_seed, transform)
    acc1, acc5        = self._evaluate_accuracy()
    if display:
      print(f"[I] Measured accuracy for the Torch model '{model.__class__.__name__}'")
      print( "    Top1=%.4f, Top5=%.4f" %(acc1, acc5))
    return  acc1, acc5


  def _load_dataset(self, images, batch_size, shuffle_seed, transform):
    dataset   = torchvision.datasets.ImageFolder(self.validate_dir, transform=transform)
    generator = torch.Generator().manual_seed(shuffle_seed) if shuffle_seed else None

    if images:
      loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(
          dataset,
          num_samples=images,
          generator=generator
        )
      )
      print(f"[I] Validate for {images} images from '{self.validate_dir}'")
    else:
      loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
      )
      print(f"[I] Validate for {self.validate_images} images from '{self.validate_dir}'")
    self._dataset = loader
