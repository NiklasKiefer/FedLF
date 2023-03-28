import flwr as fl
import tensorflow as tf
import numpy as np
from pprint import pprint as pp
from ..datahandler import get_timeseries_data

TFLITE_MODEL_PATH = 'lite_model.tflite'
TFLITE_CKPT_PATH = 'lite_model.ckpt'

scaler, train_data, test_X, test_Y = get_timeseries_data()
X_train = [x for [x,_] in train_data]
Y_train = [y for [_,y] in train_data]

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        extract = interpreter.get_signature_runner("extract")
        return extract()

    def fit(self, parameters, config):
        train_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        train_interpreter.allocate_tensors()
        restore = train_interpreter.get_signature_runner("restore")
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}