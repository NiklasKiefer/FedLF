import tensorflow as tf
from fl_model import FLModel

SAVED_MODEL_PATH = 'models/model1' # Path for saving the model in a SavedModel format before converting it to TF-Lite
TFLITE_MODEL_PATH = 'models/lite_model.tflite' # Path for saving a model as a .tflite file.

print("Creating new model...")
model = FLModel()

print("Saving new model in SavedModel format...")
tf.saved_model.save(
    model,
    SAVED_MODEL_PATH,
    signatures={
        'train':
            model.train.get_concrete_function(),
        'infer':
            model.infer.get_concrete_function(),
        'save':
            model.save.get_concrete_function(),
        'restore':
            model.restore.get_concrete_function(),
        'extract':
            model.extract.get_concrete_function()
    })

print("Convert model to tfalite model....")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

print("Saving tflite model in a .tflite file...")
open(TFLITE_MODEL_PATH, "wb").write(tflite_model)
print("Lite model was successfully saved!")