import json
import time
import pandas as pd
import os
import numpy as np
import argparse
from keras import models, Input, Model
from keras.callbacks import EarlyStopping


def load_settings():
    setting_file = open(os.path.join("assets", "settings.json"))
    settings = dict(json.load(setting_file))
    setting_file.close()
    return settings


settings = load_settings()
labels2int = {b: a for a, b in enumerate(settings["labels"])}


def argdict(defaults: dict):
    parser = argparse.ArgumentParser(prog="Stability Model Testing")
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    return args


def get_filenames(folder_path):
    return list(
        map(
            lambda y: os.path.join(folder_path, y),
            filter(lambda x: x[-4:] == ".csv", os.listdir(folder_path)),
        )
    )


def convert_df_labels(df1, labels2int):
    df = df1.copy()
    for i in range(len(df)):
        label = df["label"][i]
        df.at[i, "label"] = labels2int[label]
    return df


def split_data_with_label(df, valid_size, test_size):
    df_input = df.copy()
    df_target = df_input.pop("label")
    groups = {}
    current_group_label = None
    current_group = []
    for i, row in enumerate(df_input.itertuples(index=False)):
        if current_group_label is None:
            current_group_label = df_target[i]
        if current_group_label == df_target[i]:
            current_group.append(row)
        else:
            groups[current_group_label] = groups.get(current_group_label, [])
            groups[current_group_label].append(current_group)
            current_group_label = df_target[i]
            current_group = []
    if len(current_group):
        groups[current_group_label] = groups.get(current_group_label, [])
        groups[current_group_label].append(current_group)

    x_train, x_valid, x_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    for label, group in groups.items():
        # random.shuffle(group)
        combined = [j for i in group for j in i]
        n_test = int(len(combined) * test_size)
        n_valid = int(len(combined) * valid_size)
        n_train = len(combined) - n_test - n_valid
        for i in range(len(combined)):
            (
                x_train if i < n_train else x_valid if i < n_train + n_valid else x_test
            ).append(combined[i])
            (
                y_train if i < n_train else y_valid if i < n_train + n_valid else y_test
            ).append(label)
    return (
        np.array(x_train),
        np.array(y_train),
        np.array(x_valid),
        np.array(y_valid),
        np.array(x_test),
        np.array(y_test),
    )


def split_data_without_label(df, valid_size, test_size):
    df_input = df.copy()
    df_target = df_input.pop("label")
    x_train, x_valid, x_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    n_test = int(len(df_input) * test_size)
    n_valid = int(len(df_input) * valid_size)
    n_train = len(df_input) - n_test - n_valid
    for i, row in enumerate(df_input.itertuples(index=False)):
        (
            x_train if i < n_train else x_valid if i < n_train + n_valid else x_test
        ).append(row)
        (
            y_train if i < n_train else y_valid if i < n_train + n_valid else y_test
        ).append(df_target[i])
    return [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]


def split_data(DATA, VALID_RATIO, TEST_RATIO, index=False):
    DBs = None
    if index:
        DBs = [pd.read_csv(name, index_col=0) for name in DATA]
    else:
        DBs = [pd.read_csv(name) for name in DATA]
    DB = pd.concat(DBs, axis=0, ignore_index=True, sort=False)
    DB = convert_df_labels(DB, labels2int)

    return split_data_without_label(DB, VALID_RATIO, TEST_RATIO)


def group_data(data, group_size, target_function):
    x, y = data
    x_result = []
    y_result = []
    x_temp = []
    y_temp = []
    for a, b in zip(x, y):
        x_temp.append(a)
        y_temp.append(b)
        if len(x_temp) == group_size:
            x_result.append(x_temp)
            y_result.append(target_function((x_temp, y_temp)))
            x_temp = []
            y_temp = []

    return np.array(x_result), np.array(y_result)


def save_float_array(path, arr):
    with open(path, "w") as f:
        for num in arr:
            f.write(f"{num}\n")


class ModelOperation:
    def __init__(
        self,
        model_class,
        data,
        max_epochs=100,
        valid_ratio=0.1,
        test_ratio=0.1,
        early_stop_valid_patience=10,
        early_stop_train_patience=5,
        num_train_per_config=10,
        loss="mse",
        metrics=["mse"],
        verbose=0,
        test_data=None,
        output_name=None,
        extra="",
        preprocess=None,
    ):
        self.max_epochs = max_epochs
        self.early_stop_valid_patience = early_stop_valid_patience
        self.early_stop_train_patience = early_stop_train_patience
        self.num_train_per_config = num_train_per_config
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose

        self.counter = 0
        self.model_class = model_class
        self.base_model = model_class.model
        self.preprocess = False
        self.preprocessor = None
        self.layer_options = [None] * len(self.base_model.layers)

        # x_train, y_train, x_valid, y_valid, x_test, y_test
        self.raw_data = split_data(data, valid_ratio, test_ratio)
        self.test_data = test_data

        self.output_name = output_name
        self.extra = extra

        self.defalut_params = {
            "batchsize": 16,
            "timesteps": 32,
            "optimizer": "adam",
            "preprocess": preprocess,
            "loss": "mae",
            "metrics": "mae",
        }

        self.model = None
        self.final_data = None
        self.params = self.defalut_params
        self.history = None

    def run(self):
        raise Exception("<run> method must be defined for ModelOperation")

    def evaluate(self, model):
        pass

    def build(self):
        # Reconstruct model
        layers = self.base_model.layers
        input_shape = self.final_data[0][0].shape[1:]
        print(self.final_data[0][0].shape)
        input_layer = Input(shape=input_shape)
        current_layer = input_layer
        for i, option in enumerate(self.layer_options[1:]):
            layer = layers[i + 1]
            config = layer.get_config()
            if option is not None:
                for k, v in option.items():
                    config[k] = v
            current_layer = layer.__class__(**config)(current_layer)
        model = Model(inputs=input_layer, outputs=current_layer)
        return model

    def train(self, clean_model):
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = self.final_data
        model = models.clone_model(clean_model)
        model.compile(
            optimizer=self.params.get("optimizer"),
            loss=self.params.get("loss"),
            metrics=[self.params.get("metrics")],
        )
        batchsize = self.params.get("batchsize")
        history = model.fit(
            x_train,
            y_train,
            epochs=self.max_epochs,
            validation_data=(x_valid, y_valid),
            batch_size=batchsize,
            callbacks=[
                EarlyStopping(
                    monitor="loss",
                    patience=self.early_stop_train_patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                    start_from_epoch=8,
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stop_valid_patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                    start_from_epoch=8,
                ),
            ],
            verbose=self.verbose,
            shuffle=True,
        )
        self.epochs_record = history.history
        epochs = len(history.history["loss"])
        loss = model.evaluate(x_train, y_train, batch_size=batchsize, verbose=0)[0]
        val_loss = model.evaluate(x_valid, y_valid, batch_size=batchsize, verbose=0)[0]
        test_loss = []
        if self.test_data is not None:
            timesteps = self.params.get("timesteps")
            for test in self.test_data:
                x_test, y_test = group_data(
                    test, timesteps, self.model_class.target_function
                )
                test_loss.append(
                    model.evaluate(x_test, y_test, batch_size=batchsize, verbose=0)[0]
                )
        elif len(x_test) > 0:
            test_loss.append(
                model.evaluate(x_test, y_test, batch_size=batchsize, verbose=0)[0]
            )
        self.model = model
        self.evaluate(model)
        return epochs, loss, val_loss, test_loss


class ModelTest(ModelOperation):
    def __init__(self, model_class, data, options, *args, **kwargs):
        super().__init__(model_class=model_class, data=data, *args, **kwargs)
        self.final_options = [
            (k, (v if isinstance(v, list) else [v]))
            for k, v in options.items()
            if not (isinstance(v, list) and len(v) == 0)
        ]
        for name1, param1 in self.defalut_params.items():
            found = False
            for name2, param2 in self.final_options:
                if name1 == name2:
                    found = True
            if not found:
                self.final_options.append((name1, [param1]))
        self.current_options = [None] * len(self.final_options)

    def evaluate(self, model):
        """
        This function is used to retrieve testing losses for
        stable and unstable frames for each model trains
        """
        # timesteps = self.params.get("timesteps")
        # x_test_1, y_test_1 = group_data(self.test_data[0], timesteps, self.model_class.target_function)
        # results_1 = model.predict(x_test_1)
        # mse_arr_1 = [np.mean((v1 - v2)**2) for v1, v2 in zip(y_test_1, results_1)]
        # vector_size = list(self.params['layer1'].values())[0]
        # prefix = self.output_name+'-'+str(len(os.listdir("losses"))).zfill(2)+'-'+str(vector_size)
        # output_file = os.path.join("losses", prefix+'_1.csv')
        # save_float_array(output_file, mse_arr_1)
        # x_test_2, y_test_2 = group_data(self.test_data[1], timesteps, self.model_class.target_function)
        # results_2 = model.predict(x_test_2)
        # mse_arr_2 = [np.mean((v1 - v2)**2) for v1, v2 in zip(y_test_2, results_2)]
        # output_file = os.path.join("losses", prefix+'_2.csv')
        # save_float_array(output_file, mse_arr_2)
        pass

    def process_options(self):
        self.final_data = list(self.raw_data)
        self.params = {}
        for i in range(len(self.layer_options)):
            self.layer_options[i] = None
        for i, (name, options) in enumerate(self.final_options):
            option_idx = self.current_options[i]
            option = options[option_idx]
            if name == "preprocess" and option is not None:
                for i in range(3):
                    if self.test_data and i == 2:
                        continue
                    self.final_data[i] = option.transform(self.final_data[i])
            if name[:5] == "layer":
                layer_number = int(name[5:])
                self.layer_options[layer_number] = option
            self.params[name] = option
        timesteps = self.params.get("timesteps")
        for i in range(3):
            self.final_data[i] = group_data(
                self.final_data[i], timesteps, self.model_class.target_function
            )

    def run(self, output_path):
        self.history = []
        self.test(0)
        output_file = os.path.join(output_path, str(int(time.time())) + ".csv")
        if self.output_name:
            output_file = os.path.join(
                output_path,
                str(len(os.listdir(output_path))).zfill(2)
                + "-"
                + self.output_name
                + ".csv",
            )
        pd.DataFrame(
            data=self.history,
            columns=list(next(zip(*self.final_options)))
            + ["avg_epochs", "avg_loss", "avg_valid_loss"]
            + (["avg_test_loss"] if len(self.raw_data[2][0]) else [])
            + [f"avg_test_loss_{i}" for i, v in enumerate(self.test_data or [])],
        ).to_csv(output_file)
        with open(output_file, "a") as of:
            of.write("\n")
            of.write(self.extra)

    def test(self, option_idx):
        if option_idx == len(self.final_options):
            return self.build_and_train()
        name, options = self.final_options[option_idx]
        for i, v in enumerate(options):
            self.current_options[option_idx] = i
            self.test(option_idx + 1)

    def build_and_train(self):
        self.process_options()
        print("=================================================================")
        [
            print(f"{name:12}: {self.params.get(name) or 'No Change'}")
            for name in self.params.keys()
        ]
        print()
        model = self.build()
        model.summary()
        train_results = []
        labels = ["round", "epochs", "train", "valid", "test"]
        print("{:>8} {:>8} {:>8} {:>8} {:>8}".format(*labels))
        for i in range(self.num_train_per_config):
            record = self.train(model)
            record = list(record[:-1]) + list(record[-1])
            train_results.append(record)
            print("{:8} {:8.0f} {:8.4f} {:8.4f} {:8.4f}".format(i, *record))
        record = [sum(i) / len(i) for i in zip(*train_results)]
        print(("{:>8} {:8.0f}" + " {:8.4f}" * (len(record) - 1)).format("avg", *record))
        self.history.append(
            [self.params.get(name) or "No Change" for name in self.params.keys()]
            + record
        )
        print("-----------------------------------------------------------------\n")
