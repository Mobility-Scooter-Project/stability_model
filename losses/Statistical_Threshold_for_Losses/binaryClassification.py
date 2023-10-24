import pandas as pd
import numpy as np
from prettytable import PrettyTable


def classify_value(value, upperLimit):
    """
    Classify value as "stable" or "unstable" based on limits.
    To be used in conjuction with applymap() function.

    Parameters:
    value (int or float): Value to compare to boundaries
    upperLimit (int or float): Upper bound of value

    Return:
    str: Predicted classification
    """
    
    if(value > upperLimit):
        return "unstable"
    else:
        return "stable" 


def confusion_matrix(tp, fp, fn, tn):
    """
    Return a neatly formatted confusion matrix for printing or writing to a file

    Parameters: 
    tp (number): true positive
    fp (number): false positive
    fn (number): false negative
    tn (number): true negative

    Return: 
    str: Confusion matrix as a string
    """


    conf_mat = PrettyTable(['','Predicted Positive', 'Predicted Negative'])
    conf_mat.add_row(['Actual Positive', tp, fp])
    conf_mat.add_row(['Actual Negative', fn, tn])
    return conf_mat.get_string()


def testThresholds(trainingDataList, numOfSD, stableData, unstableData, outputFile, modelName):
    """
    Classify stable and unstable test sets as "stable" or "unstable".
    Model is trained on average loss and standard deviations of training data.
    Output results to a text file.

    Parameters: 
    trainingDataList (str array): Array of names of training data files
    numOfSD (int): Desired number of standard deviations to set as upper threshold
    stableData (str): Name of stable test data file
    unstableData (str): Name of unstable test data file
    outputFile (str): Desired name of output file for results
    modelName (str): Name of model and desired name for results in output file
    """


    # Iterate through list of training datasets to combine all into one large training dataframe
    tempDFList = []
    for fileName in trainingDataList:
        tempDF = pd.read_csv(fileName, header=None)
        tempDFList.append(tempDF)
    fullTrainingDF = pd.concat(tempDFList, ignore_index=True)

    # Calculate average loss
    avgLoss = fullTrainingDF.mean()

    # Calculate standard deviation
    stanDev = fullTrainingDF.std()

    # Set upper threshold 
    #   based on xx number of standard deviations above the mean
    upperBound = avgLoss + (numOfSD * stanDev)
    upperBound = float(upperBound)


    # PREDICT CLASSIFICATIONS
    # Read the stable and unstable test datasets
    test_stableDF = pd.read_csv(stableData, header=None)
    test_unstableDF = pd.read_csv(unstableData, header=None)
    
    # Classify stable test set
    test_stableDF['Classifications'] = test_stableDF.applymap(lambda x: classify_value(x, upperBound))
    # FN = num of unstable classifications for stable test set
    FN = test_stableDF['Classifications'].value_counts().get('unstable', 0)

    # Classify unstable test set
    test_unstableDF['Classifications'] = test_unstableDF.applymap(lambda x: classify_value(x, upperBound))
    # FP = num of stable classes for UNstable test set
    FP = test_unstableDF['Classifications'].value_counts().get('stable', 0)

    # Get total number of true stable labels from size of stable test set
    numOfTrueStable = test_stableDF.shape[0]
    # Get total number of true unstable labels from size of unstable test set
    numOfTrueUnstable = test_unstableDF.shape[0]

    # Calc TP and TN from number of true labels - FP or FN
    TP = numOfTrueStable - FN
    TN = numOfTrueUnstable - FP

    # Calc accuracy, precision, and recall
    accuracy = (TP + TN)/(TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Save results to text file
    with open(outputFile, 'w') as file:
        file.write(modelName + " Binary Classification\n")
        file.write("Upper threshold is " + str(numOfSD) + " standard deviations above the average loss\n")
        file.write(confusion_matrix(TP, FP, FN, TN) + "\n")
        file.write("Accuracy: " + str(accuracy) + "\n")
        file.write("Precision: " + str(precision) + "\n")
        file.write("Recall: " + str(recall) + "\n")

def main():
    # Classify Model 1 Data
    trainingArr_m1 = ["./losses/model1_train.csv", "./losses/model1_valid.csv"]
    testThresholds(trainingArr_m1, 2, "./losses/model1_test_stable.csv", "./losses/model1_test_unstable.csv", "Model1_BC_2SD.txt", "Model 1")

    # Classify Model 2 Data
    trainingArr_m2 = ["./losses/model2_train.csv", "./losses/model2_valid.csv"]
    testThresholds(trainingArr_m2, 2, "./losses/model2_test_stable.csv", "./losses/model2_test_unstable.csv", "Model2_BC_2SD.txt", "Model 2")


if __name__ == "__main__":
    main()
