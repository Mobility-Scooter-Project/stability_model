'''
lstm -> repeat vector -> lstm
'''

from keras import Input, layers, Model
TIMESTEPS = 16
NUM_FEATURES = 27
VECTOR_SIZE = 10
class Encoder_Decoder:
    inputs = Input(shape=(TIMESTEPS, NUM_FEATURES))
    lstm = layers.LSTM(VECTOR_SIZE, return_sequences=False)(inputs)
    repeat = layers.RepeatVector(TIMESTEPS)(lstm)
    outputs = layers.LSTM(NUM_FEATURES, return_sequences=True)(repeat)
    model = Model(inputs=inputs, outputs=outputs)

    def target_function(arr):
        return arr
