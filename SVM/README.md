# Support Vector Machine

Classify 2D Movenet pose data as "stable" or "unstable".

## Description

This code compiles all data from 2D Movenet files with a keyword in their name to train/test a support vector machine. Only a portion of the Movenet data is used, as determined by the samplingRate variable. Data is split between training and testing. Data is grouped in rows of &lt;frameSeq&gt;, flattened column-wise, and used to train and test the SVM.  
  
The training program conducts a grid search with stratified k-folding to find the best hyperparameters and frameSeq-samplingRate combination to produce an SVM with the highest recall score. It outputs a trained SVM joblib as well as the frame sequence and sampling rate required to run the SVM in real-world applications. It also outputs the anamoly metrics of training/testing accuracies, recall, precision, F1, MCC (Matthews Correlation Coefficient), and AP (area under the precision-recall curve (AUC-PR)) for each possible frameSequence-samplingRate combo and their respective set of best SVM hyperparameters. Further, it saves the best hyperparameters for each frameSequence-samplingRate combo (used internally during training).  
  
The testing program outputs a hashmap with the flattened coordinate groups and their classifications. This program is intended to integrate with the backend of the mobile app.

## Getting Started

### Dependencies

* Python3
* scikit-learn
* numPy
* pandas
* joblib
* imbalanced-learn

To download Python libraries:
```
pip install -r requirements.txt
```

## Executing program

### Training
In svm_training.py main():
1. Change dataDirectory to the folder with training data CSV files.
2. (Optional) Change fileKeyword; pull data from file names with matching keyword
3. (Optional) Change testingRatio to desired ratio for testing data; testing data will get &lt;testingRatio&gt; portion of data, training data will get &lt;1-testingRatio&gt;.
4. (Optional) Change iterations to number of splits on which to test the SVM and calculate average anamoly metrics. More iterations could improve cross-validation and representation of anamoly metrics but default of 10 should be sufficient.
5. (Optional) Change jobFileName to desired name of output SVM joblib file
6. Run svm_training.py. Use svm joblib file and frameSequence-samplingRate.txt for real-world applications and testing. ***May take 45 minutes to run, depending on computational power of your computer.**

### Testing
Use svm_testing.py
1. The model can be used in the main() function of svm_testing.py OR imported to another python program:
```
import svm_testing as svm

svm.classify_pose_data(...)
```
2. Input the coordinates array, path to the SVM job file, and path to frameSeq-samplingRate.txt to the classify_pose_data() function.
3. Run svm_testing.py OR your python program. Code returns a hashmap with the coordinate groups and their corresponding classifications.

## Notes and Potential Future Improvements
The data used to train the SVM was highly imbalanced (97% stable, 3% unstable). Thus, identifying the minority "unstable" class is a form of anamoly detection and recall is the basis for determining the the "best" SVM. Accuracy is not used to gauge the best SVM since overfitting and high class imbalance could result in misleadingly high accuracy. (That is, the SVM could guess 100% of testing samples are stable and it would be 97% "accurate" but this is not a good representation of performance.)  
  
Further, this imbalance makes it difficult to generalize the data and avoid overfitting, leading to variance in results. Consequentially, rerunning the training program multiple times (even though the SVM is tested against 10 different splits internally to calculate average metrics) may output different hyperparameters, frameSequence, samplingRate, and anamoly metrics each time. Thus, in the future, the following data augmentation could potentially improve the generalization of the SVM and its recall score:
1. Resampling the data before feeding it the SVM, such as oversampling the minority class and/or undersampling the majority class. ***class_weight='balanced' parameter in SVC() calls MUST be deleted or set to None. Else, overcompensation and distorted results may occur.***
2. Normalize the input coordinates; each driver may start at different coordinates due to camera or torso positioning, etc. It may be better to capture the change in coordinates over time, rather than the absolute values of the coordinates. 
3. Gather more data, especially "unstable" samples.
