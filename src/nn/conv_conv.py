from keras import Input, layers, Model

TIMESTEPS = 64
VECTOR_SIZE = 10

class Encoder_Decoder:
    def __init__(self, number_of_features):
        inputs = Input(shape=(TIMESTEPS, number_of_features))
        x = layers.Conv1D(1, 3, padding="same")(inputs)
        outputs = layers.Conv1DTranspose(number_of_features, 3, padding="same")(x)
        self.model = Model(inputs=inputs, outputs=outputs)

    def target_function(self, data):
        x, y = data
        return x


OPTIONS = {
    "batchsize": [40],
    "timesteps": [TIMESTEPS],
    "optimizer": ["adam"],
    "loss": ['mse'],
    "metrics": ['mse'],
    "layer1": [{"filters": i} for i in [1, 2, 4, 8]],
}
