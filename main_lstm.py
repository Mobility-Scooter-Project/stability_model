import time
import pandas as pd
import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from vutils import load_settings
from mutils import split_data, group_data, get_filenames
from keras import Input, Model, layers
from keras.callbacks import EarlyStopping

settings = load_settings()
labels2int = {b: a for a, b in enumerate(settings["labels"])}

landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]

DATA = get_filenames("data")
MAX_EPOCHS = 30
max_epochs=100,
valid_ratio=0.1,
test_ratio=0.1,
early_stop_valid_patience=10,
early_stop_train_patience=5,
num_train_per_config=10,
loss='mse',
metrics=['mse'],
verbose=0,
test_data=None
final_data = split_data(data, valid_ratio, test_ratio)

optimizer="adam"
batchsize=16
timestamp= 32
preprocess = None
input_shape = final_data[0][0].shape[1:]

# LSTM model construction
inputs = Input(shape=input_shape)
lstm = layers.LSTM(256, return_sequences=True)(inputs)
lstm = layers.LSTM(32)(lstm)
outputs = layers.Dense(3, activation="softmax")(lstm)
model = Model(inputs=inputs, outputs=outputs)

model.summary()

# training
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = final_data
model.compile(
    optimizer=optimizer, loss=loss, metrics=metrics
)
batchsize = batchsize
history = model.fit(
    x_train,
    y_train,
    epochs=max_epochs,
    validation_data=(x_valid, y_valid),
    batch_size=batchsize,
    callbacks=[
        EarlyStopping(
            monitor="loss",
            patience=early_stop_train_patience,
            restore_best_weights=True,
            verbose=verbose,
            start_from_epoch=8,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stop_valid_patience,
            restore_best_weights=True,
            verbose=verbose,
            start_from_epoch=8,
        ),
    ],
    verbose=verbose,
    shuffle=False,
)
epochs_record = history.history
epochs = len(history.history["loss"])
loss = model.evaluate(x_train, y_train, batch_size=batchsize, verbose=0)[0]
val_loss = model.evaluate(x_valid, y_valid, batch_size=batchsize, verbose=0)[0]
test_loss = []
if test_data is not None:
    timestamp = timestamp
    for test in test_data:
        x_test, y_test = group_data(test, timestamp, self.model_class.target_function)
        test_loss.append(model.evaluate(x_test, y_test, batch_size=batchsize, verbose=0)[0])
elif len(x_test)>0:
    test_loss.append(model.evaluate(x_test, y_test, batch_size=batchsize, verbose=0)[0])
