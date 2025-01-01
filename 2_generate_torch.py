import keras
from src.framework.keras2rex import Keras2RexGenerator

if __name__=="__main__":
  model_name = "mobilenetv1"
  model      = keras.models.load_model(f"output/{model_name}_opt.keras")
  generator  = Keras2RexGenerator(model, 'mobilenetv1', True)
  generator.create_trex_model(
    output_dir="output"
    , source_file=f"{model_name}.py"
    , weight_file=f"{model_name}.pt"
  )
