from keras import Input, layers, Model

TIMESTEPS = 16
NUM_FEATURES = 27
VECTOR_SIZE = 10
class Encoder_Decoder:
    inputs = Input(shape=(TIMESTEPS, NUM_FEATURES))
    lstm = layers.LSTM(64, return_sequences=True)(inputs)
    lstm = layers.LSTM(32, return_sequences=True)(lstm)
    lstm = layers.LSTM(16, return_sequences=True)(lstm)
    lstm = layers.LSTM(32, return_sequences=True)(lstm)
    lstm = layers.LSTM(64, return_sequences=True)(lstm)
    outputs = layers.LSTM(NUM_FEATURES, return_sequences=True)(lstm)
    model = Model(inputs=inputs, outputs=outputs)

    def target_function(arr):
        return arr