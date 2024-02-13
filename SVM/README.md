# Support Vector Machine

Classify 2D Movenet pose data as "stable" or "unstable".

## Description

This code compiles all data from 2D Movenet files with a keyword in their name to train/test a support vector machine. Only a portion of the Movenet data is used, as determined by the samplingRate variable. Data is split between training and testing. Data is grouped in rows of &lt;frameSeq&gt;, flattened column-wise, and used to train and test the SVM. The training program outputs the training and testing accuracies. The testing program outputs a hashmap with the flattened groups and their classifications.

## Getting Started

### Dependencies

* Python3
* scikit-learn
* numPy
* pandas
* joblib

To download Python libraries:
```
pip install -r requirements.txt
```

## Executing program

### Training
In svm_training.py main():
1. Change dataDir to location of training and testing data.
2. (Optional) Change fileKeyword; use the data from all the files with the keyword.
3. (Optional) Change samplingRate to desired rate; gets every &lt;samplingRate&gt; piece of data (ex: every 4th data point).
4. (Optional) Change frameSeq to desired frame sequence length; groups rows of &lt;frameSeq&gt; for input to SVM.
5. (Optional) Change testingRatio to desired ratio for testing data; testing data will get &lt;testingRatio&gt; portion of data, training data will get &lt;1-testingRatio&gt;.
6. Run svm_training.py. Code outputs the training and testing accuracies to the console and a text file within a "Results" folder. It also outputs an SVM job file as well as a text file with the frameSeq and samplingRate parameters. These files are exported to use the SVM in any program.

### Testing
Use svm_testing.py
1. The model can be used in the main() function of svm_testing.py OR imported to another python program:
```
import svm_testing as svm

svm.classify_pose_data(...)
```
2. Input the coordinates array, the path to the SVM job file, and the path to the parameters file (with the frameSeq and samplingRate) to the classify_pose_data() function.
3. Run svm_testing.py OR your python program. Code returns a hashmap with the coordinate groups and their corresponding classifications.
