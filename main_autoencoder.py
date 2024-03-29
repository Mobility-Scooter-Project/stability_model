#!/data03/home/ruoqihuang/anaconda3/envs/tf/bin/python
import os
import importlib
from src.mutils import ModelTest, split_data, get_filenames, argdict
from src.preprocessor import StableFilter, UnstableFilter


args = argdict(
    {
        "model": "conv_conv",
        "max_epochs": 40,
        "valid_ratio": 0.3,
        "num_train_per_config": 1,
        "verbose": 0,
    }
)

autoencoder = importlib.import_module(f"src.nn.{args.model}")
Encoder_Decoder, OPTIONS = autoencoder.Encoder_Decoder, autoencoder.OPTIONS
NN_NAME = args.model
MAX_EPOCHS = args.max_epochs
DATA3D = get_filenames("3d_data")
TEST_DATA3D = get_filenames("3d_test_data")
DATA2D = get_filenames("2d_data")
TEST_DATA2D = get_filenames("2d_test_data")


stable_test_data_3d = StableFilter(stable_label=0, padding=30).transform(
    split_data(TEST_DATA3D, 0, 0, index=True)[0]
)
unstable_test_data_3d = UnstableFilter(stable_label=0, padding=10).transform(
    split_data(TEST_DATA3D, 0, 0, index=True)[0]
)
stable_test_data_2d = StableFilter(stable_label=0, padding=30).transform(
    split_data(TEST_DATA2D, 0, 0)[0]
)
unstable_test_data_2d = UnstableFilter(stable_label=0, padding=10).transform(
    split_data(TEST_DATA2D, 0, 0)[0]
)
extra = ""
with open(os.path.join("src", "nn", f"{NN_NAME}.py")) as nnf:
    extra += nnf.read()


SETTINGS_2D = {
    "max_epochs": MAX_EPOCHS,
    "valid_ratio": args.valid_ratio,
    "test_ratio": 0,
    "early_stop_valid_patience": MAX_EPOCHS // 10,
    "early_stop_train_patience": MAX_EPOCHS // 10,
    "num_train_per_config": args.num_train_per_config,
    "verbose": args.verbose,
    "test_data": [unstable_test_data_2d, stable_test_data_2d],
    "output_name": NN_NAME,
    "extra": extra,
}

ModelTest(Encoder_Decoder(18), DATA2D, OPTIONS, **SETTINGS_2D).run(
    output_path="2d_test_results"
)

"""
Commented out since 3D pose estimation is worse due to occlusion
"""
# SETTINGS_3D = {
#     "preprocess": StableFilter(stable_label=0, padding=30),
#     "max_epochs":MAX_EPOCHS,
#     "valid_ratio":0.3,
#     "test_ratio":0,
#     "early_stop_valid_patience":MAX_EPOCHS//10,
#     "early_stop_train_patience":MAX_EPOCHS//10,
#     "num_train_per_config":3,
#     "verbose": 0,
#     "test_data": [unstable_test_data_3d, stable_test_data_3d],
#     "output_name": NN_NAME,
#     "extra": extra
# }

# ModelTest(Encoder_Decoder(27), DATA3D, OPTIONS, **SETTINGS_3D).run(output_path="3d_test_results")
