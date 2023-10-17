# Stability Model
## Setup
- Install Anaconda
- Google how to add `conda` to path if not recognized
```sh
conda activate
conda create -n tf python=3.9 -y
conda activate tf
pip install -r requirements.txt
```

## Usage
- Training:
```sh
conda activate tf
python main_autoencoder.py --model <model_file>
```
  - You will need `2d_data`, `2d_test_data`, `3d_data`, and `3d_test_data` folders under the `stability_model` folder
    - `XX_test_data` and `XX_data` currently have different format
  - You will need `2d_test_results` and `3d_test_results` folders under the `stability_model` folder to save test results
  - Model file is located at `src/nn`
  - Planning on more args and eventually have all params as args


## Documation
* `mutils.py` contains the actual operation code for model training and testing
  * `ModelTest` takes:
    * model class in `src/nn`
    * `OPTIONS: dict[str:list]` that provides different values to test the model
    * `SETTINGS: dict` to configure the pipeline details for all testing
    

## Visitor Count
![Visitor Count](https://profile-counter.glitch.me/huangruoqi/count.svg)
