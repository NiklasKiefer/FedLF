import tensorflow as tf
import numpy as np
import pprint as pp

TFLITE_MODEL_PATH = 'models/lite_model.tflite'
TFLITE_CKPT_PATH = 'models/ckpts/lite_model.ckpt'

def dict_dims(mydict):
    d1 = len(mydict)
    d2 = 0
    for d in mydict:
        d2 = max(d2, len(d))
    return d1, d2


interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

extract = interpreter.get_signature_runner("extract")
weights = extract()
for val in weights:
    print(weights[val].shape)

# What I want to do now:
#init_weights = interpreter.get_signature_runner("init_weights")
#init_weights(weights)