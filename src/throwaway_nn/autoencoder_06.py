'''
conv1d -> repeat vector -> lstm
'''

from keras import Input, layers, Model
TIMESTEPS = 64
VECTOR_SIZE = 10
class Encoder_Decoder:
    def __init__(self, number_of_features):
        inputs = Input(shape=(TIMESTEPS, number_of_features))
        x = layers.Conv1D(1, 3, padding="same")(inputs)
        x = layers.Flatten()(x)
        repeat = layers.RepeatVector(TIMESTEPS)(x)
        outputs = layers.LSTM(number_of_features, return_sequences=True)(repeat)
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
    "layer1": [{"filters": i} for i in [5, 10, 20, 40]],
}