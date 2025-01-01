import torch
import argparse
from output.mobilenetv1        import Mobilenetv1
from torchvision.transforms    import transforms
from vision.imagenetclassifier import ImagenetVerifierFactory

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Generate Keras model')
  parser.add_argument('--datadir', default="./data", type=str, help="dataset path")
  args = parser.parse_args()

  model_gen   = Mobilenetv1() 
  model_gen.load_state_dict(torch.load('output/mobilenetv1.pt'))

  ImagenetVerifierFactory(args.datadir, framework="torch").evaluate(
    model_gen,
    transform=transforms.Compose([
      transforms.Resize(size=256),
      transforms.CenterCrop(size=224),
      transforms.ToTensor(),
      transforms.Normalize(mean=0.5, std=0.5)
    ]),
    shuffle_seed=1234,
    images=1000
  )

