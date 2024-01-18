# Statistics-Based Binary Classifier

Classify loss data as "stable" or "unstable".

## Description

This code classifies loss data as "stable" or "unstable" based on a statisical threshold. It finds the average loss value and determines an upper threshold based on a specified number of standard deviations above the mean. All loss values above the threshold are classified as "unstable" and all values at or below the threshold as "stable". The output is a confusion matrix with true positive, false positive, true negative, false negative, accuracy, precision, and recall values. The data, classification, and output files are different for models 1 and 2.

## Getting Started

### Dependencies

* Python3
* numPy
* pandas
* prettytable
To download Python libraries:
```
pip install -r requirements.txt
```

### Executing program
In main():
1. List the directories of model 1 training data in trainingArr_m1[]; can input 1 or more file names.
2. List the directories of model 2 training data in trainingArr_m2[].
3. Change test_stable_m1 and test_stable_m2 to the directories of the testing data with "stable" labels for model 1 and model 2, respectively.
4. Change test_unstable_m1 and test_unstable_m2 to the directories of the testing data with "unstable" labels for model 1 and model 2, respectively.
5. (Optional) standardDeviations determines the number of standard deviations above the mean to classify data as "unstable". You can change this value to increase or decrease the upper threshold. 2 standard deviations have already been chosen as a start.
6. Run binClass_statsThreshold.py. Code outputs confusion matrices to files, one for each model.