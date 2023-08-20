from nn.lstm import LSTM
from mutils import ModelTest, split_data, get_filenames
from preprocessor import Balancer



MAX_EPOCHS = 1

DATA = get_filenames("data")
print(DATA)

OPTIONS = {
    "preprocess": [Balancer(100, 50)],
    "batchsize": [40],
    "timestamp": [16],
    "optimizer": ["adam"],
    "layer1": [{"units": i*5} for i in range(1, 10)],
}

SETTINGS = {
    "max_epochs":MAX_EPOCHS,
    "valid_ratio":0.3,
    "test_ratio":0,
    "early_stop_valid_patience":MAX_EPOCHS//10,
    "early_stop_train_patience":MAX_EPOCHS//10,
    "num_train_per_config":10,
    "loss":"sparse_categorical_crossentropy",
    "metrics": ['accuracy'],
    "verbose": 1,
}

ModelTest(LSTM, DATA, OPTIONS, **SETTINGS).run()