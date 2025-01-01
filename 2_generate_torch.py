import keras
from src.framework.keras2rex import Keras2RexGenerator

if __name__=="__main__":
  for model_name in [
    "mobilenet1",
    "mobilenet2",
  ]:
    model      = keras.models.load_model(f"output/{model_name}_opt.keras")
    generator  = Keras2RexGenerator(model, model_name, True)
    generator.create_trex_model(
      output_dir="output"
      , source_file=f"{model_name}.py"
      , weight_file=f"{model_name}.pt"
    )
