from keras import Input, layers, Model

TIMESTEPS = 16
VECTOR_SIZE = 10
class Encoder_Decoder:
    def __init__(self, number_of_features):
        inputs = Input(shape=(TIMESTEPS, number_of_features))
        lstm = layers.LSTM(VECTOR_SIZE, return_sequences=False)(inputs)
        dense = layers.Dense(VECTOR_SIZE)(lstm)
        repeat = layers.RepeatVector(TIMESTEPS)(dense)
        outputs = layers.LSTM(number_of_features, return_sequences=True)(repeat)
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
    # "layer1": [{"units": i*5} for i in range(1, 10)],
}