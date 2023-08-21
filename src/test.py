#!/data03/home/ruoqihuang/anaconda3/envs/tf/bin/python
from mutils import ModelTest, split_data, get_filenames
from preprocessor import StableFilter, UnstableFilter

from nn.autoencoder_00 import Encoder_Decoder

OUTPUT_NAME = 'autoencoder_00'


MAX_EPOCHS = 30
DATA = get_filenames("data")
OPTIONS = {
    "preprocess": [StableFilter(stable_label=0, padding=30)],
    "batchsize": [20, 40],
    "timesteps": [16*i for i in range(1, 5)],
    "optimizer": ["adam"],
    "layer1": [{"units": i*5} for i in range(1, 10)],
}

number_of_test_file = 4
stable_test_data = StableFilter(stable_label=0, padding=30).transform(split_data(DATA[:number_of_test_file], 0, 0)[0])
unstable_test_data = UnstableFilter(stable_label=0, padding=10).transform(split_data(DATA[:number_of_test_file], 0, 0)[0])

SETTINGS = {
    "max_epochs":MAX_EPOCHS,
    "valid_ratio":0.3,
    "test_ratio":0,
    "early_stop_valid_patience":MAX_EPOCHS//10,
    "early_stop_train_patience":MAX_EPOCHS//10,
    "num_train_per_config":10,
    "loss":'mae',
    "metrics": ['mae'],
    # "loss":"sparse_categorical_crossentropy",
    # "metrics": ['accuracy'],
    "verbose": 1,
    "test_data": [unstable_test_data, stable_test_data],
    "output_name": OUTPUT_NAME
}

ModelTest(Encoder_Decoder, DATA[number_of_test_file:], OPTIONS, **SETTINGS).run()