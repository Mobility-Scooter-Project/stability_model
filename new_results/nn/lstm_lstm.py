
from keras import Input, layers, Model

TIMESTEPS = 16
VECTOR_SIZE = 3
class Encoder_Decoder:
    def __init__(self, number_of_features):
        inputs = Input(shape=(TIMESTEPS, number_of_features))
        x = layers.LSTM(VECTOR_SIZE, return_sequences=True)(inputs)
        outputs = layers.LSTM(number_of_features, return_sequences=True)(x)
        self.model = Model(inputs=inputs, outputs=outputs)

    def target_function(self, data):
        x, y = data
        return x

OPTIONS = {
    "batchsize": [40],
    "timesteps": [TIMESTEPS],
    "optimizer": ["adam"],
    "loss": ['mse', 'mae'],
    "metrics": ['mse'],
    "layer1": [{"units": i} for i in [1, 2, 4, 8]],
}