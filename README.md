# Stability Model
## TODO
* self-supervised learning multimodal pipeline 
  * https://arxiv.org/pdf/2304.01008.pdf
    * pseudo-label
    * loss function

## Usage
* `pip install pandas tensorflow opencv-python pims av tqdm`
* `make test` to test different configurations specify in `src/test.py`
  * models are specified in the `src/nn` folder


## Documation
* `mutils.py` contains the actual operation code for model training and testing
  * `ModelTest` takes:
    * model class in `src/nn`
    * `OPTIONS: dict[str:list]` that provides different values to test the model
    * `SETTINGS: dict` to configure the pipeline details for all testing
  * `ModelTrain` takes:
    * model class in `src/nn`
    * `OPTIONS: dict` that provides one value to train the model
    * `SETTINGS: dict` that configure the pipeline details for training
    

## Visitor Count
![Visitor Count](https://profile-counter.glitch.me/huangruoqi/count.svg)
