# Support Vector Machine

Classify 2D Movenet pose data as "stable" or "unstable".

## Description

This code compiles all data from 2D Movenet files with a keyword in their name to train/test a support vector machine. Only a portion of the Movenet data is used, as determined by the samplingRate variable. Data is split between training and testing. Data is grouped in rows of <frameSeq>, flattened column-wise, and used to train and test the SVM. The outputs are the training and testing accuracies.

## Getting Started

### Dependencies

* Python3
* scikit-learn
* numPy
* pandas
To download Python libraries:
```
pip install -r requirements.txt
```

### Executing program
In main.py:
1. Change dataDir to location of training and testing data.
2. (Optional) Change fileKeyword; use the data from all the files with the keyword.
3. (Optional) Change samplingRate to desired rate; gets every <samplingRate> piece of data (ex: every 5th data point)
4. (Optional) Change frameSeq to desired frame sequence length; groups rows of <frameSeq> for input to SVM.
5. (Optional) Change testingRatio to desired ratio for testing data; testing data will get <testingRatio> portion of data, training data will get <1-testingRatio>.
6. (Optional) Best parameters (based on highest testing accuracy) are printed to the console and written to a text file in a "ParameterOptions" folder. Uncomment the following line to find the best SVM parameters:
```
svm.find_best_SVM(x_train_t, y_train_t, x_test_t, y_test_t, seed=10, nameExt="_temporal_"+str(frameSeq))
``` 
7. (Optional) Based on output of find_best_SVM(), use best parameters for create_svm(). Best parameters for frameSeq=64 have already been found and inputted into the create_svm() function. 
8. Run main.py. Code outputs the training and testing accuracies to the console and a text file within a "Results" folder. 