import torch
import argparse
from output.mobilenetv1        import Mobilenetv1
from output.mobilenetv2        import Mobilenetv2
from torchvision.transforms    import transforms
from src.vision.imagenetclassifier import ImagenetVerifierFactory

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Generate Keras model')
  parser.add_argument('--datadir', default="./data", type=str, help="dataset path")
  args = parser.parse_args()

  for model_name, model_gen in [
    ["mobilenetv1", Mobilenetv1()],
    ["mobilenetv2", Mobilenetv2()],
  ]:
    model_gen.load_state_dict(torch.load(f'output/{model_name}.pt'))
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

