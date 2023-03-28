import tensorflow as tf

# This class creates a sequential model with 4 signatures, being train, infer, save and restore
# The class can be used to create a .tflite model file which can then be loaded using the tf.lite.interpreter 
# After loading the file in the interpreter, the 4 signatures can be used to train and infer the tflite model.
class FLModel(tf.Module):
    WINDOW_SIZE = 4
    LEARNING_RATE = 0.0001


    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer((self.WINDOW_SIZE, 1)),
            tf.keras.layers.Conv1D(5, 4),
            tf.keras.layers.Dense(8, 'relu'),
            tf.keras.layers.Dense(1, 'linear')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss=tf.keras.losses.MeanSquaredError())

    @tf.function(input_signature=[
        tf.TensorSpec([None, WINDOW_SIZE, 1], tf.float32),
        tf.TensorSpec([None, 1], tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x)
            loss = self.model.loss(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        return result
    
    @tf.function(input_signature=[
        tf.TensorSpec([None, WINDOW_SIZE, 1], tf.float32),
    ])
    def infer(self, x):
        logits = self.model(x)
        return {
            "output": logits
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return tensors_to_save
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors
    
    @tf.function
    def extract(self):
        tmp_dict = {}
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        for index, layer in enumerate(tensors_to_save):
            tmp_dict[tensor_names[index]] = layer
        return tmp_dict
        
    #@tf.function
    # def init_weights(self, weights):
        # TODO


