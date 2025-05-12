# Statistics-Based Binary Classifier

Classify loss data as "stable" or "unstable".

## Description

This code classifies loss data as "stable" or "unstable" based on a statistical threshold. The training program finds the average loss value and determines an upper threshold based on a specified number of standard deviations above the mean. All loss values above the threshold are classified as "unstable" and all values at or below the threshold as "stable". The output of the training code is a file with the threshold as well as a confusion matrix with true positive, false positive, true negative, false negative, accuracy, precision, recall, and F1 values. The input data, classification, and output files are different for models 1 and 2.  
  
The testing program accepts an array of losses and outputs a hashmap with the losses and their classifications.

## Getting Started

### Dependencies

* Python3
* pandas
* prettytable
* json

To download Python libraries:
```
pip install -r requirements.txt
```

## Executing the programs

### Training
In stats_model_training.py main():
1. List the directories of model 1 training data in trainingArr_m1[]; can input 1 or more file names.
2. List the directories of model 2 training data in trainingArr_m2[].
3. Change test_stable_m1 and test_stable_m2 to the directories of the testing data with "stable" labels for model 1 and model 2, respectively.
4. Change test_unstable_m1 and test_unstable_m2 to the directories of the testing data with "unstable" labels for models 1 and 2, respectively.
5. (Optional) standardDeviations determines the number of standard deviations above the mean to classify data as "unstable". You can change this value to increase or decrease the upper threshold. 0 standard deviations is the default as it results in the highest recall so far.
6. Run stats_model_training.py. Code outputs text files with the thresholds for each model. It also outputs confusion matrices for each model in the "Results" folder with "_#SD" in the file name to specify the number of standard deviations used.

### Testing
Use stats_model_testing.py
1. The model can be used in the main() function of stats_model_testing.py OR imported to another python file:
```
import stats_model_testing as stats

stats.classify_losses(...)
```
2. Input the losses array and the path to the threshold file to the classify_losses() function.
3. Run stats_model_testing.py OR your python file. Code returns a hashmap with the losses and their corresponding classifications.
