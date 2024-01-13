from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# LATER: normalize coordinates
#   for each value, find the smallest coordinates and normalize them as 0
#   for other coordinates, subtract from smallest coor
# may need to balance classes w/ augmentation

# POSTPROCESSING
# use best params to classify test data and format output for mobile app display


# PREPROCESSING
def read_data(dir, samplingRate, keyword):
    """
    Return a dataframe with all relevant CSV's concatenated together.
    
    Parameters:
    dir (str): name of directory with all the CSV's
    samplingRate (number): get every nth record in a CSV to reduce number of samples
    keyword (str): find CSV files with the keyword in their names

    Return:
    DataFrame: dataframe with all of the relevant data for the machine learning model
    """
    fullDF = pd.DataFrame()
    # Iterate through all front view CSV's and concatenate into 1 giant DF
    for folder, subfolder, files in os.walk(dir):
        for fileName in files:
            if keyword in fileName:
                #print(f'File name: {fileName}')
                filePath = os.path.join(folder, fileName)
                tempDF = pd.read_csv(filePath, skiprows=samplingRate)
                fullDF = pd.concat([fullDF, tempDF], axis=0, ignore_index=True)

    # Remove any rows with label "Unlabeled"
    fullDF = fullDF[fullDF.label != 'Unlabeled']

    # Convert class labels to binary w/ label mapping
    encoding = {'Stable': 1, 'Minimum Sway': 0, 'Sway rq UE sup': 0}
    fullDF['label'] = fullDF['label'].replace(encoding)
    fullDF = fullDF.reset_index(drop=True)

    return fullDF


def split_data(df, testingRatio):
    """
    Split training/testing data based on split ratio

    Parameters:
    df (DataFrame): dataframe to split
    testingRatio (number): the fraction of data to set aside for testing

    Return:
    trainingDF (DataFrame): dataframe w/ training data
    testingDF (DataFrame): dataframe w/ testing data
    """

    trainingDF, testingDF = train_test_split(df, test_size=testingRatio, shuffle=False)
    # trainingDF.to_csv("training.csv", index=False)
    # testingDF.to_csv("testing.csv", index=False)
    return trainingDF, testingDF


def insert_rows(df, listOfIndices, frameSeq):
    """
    Expand a dataframe to become a multiple of <frameSeq> by inserting rows
        Last row in a group with the same class (an index in listOfIndices) will be copied enough times to complete the group of <frameSeq>

    Parameters:
    df (DataFrame): dataframe to expand
    listOfIndices (list): list of indices where class label changes; an index is the LAST row in a group with the same class label
    frameSeq (number): length of desired frame sequence (i.e. 64 frames)

    Return:
    df (DataFrame): Expanded df; same as passed df parameter
    """
    
    # Counter to track how row insertion affects all subsequent indices
    rowIncreaseCounter = 0
    
    for index in listOfIndices:
        # newIndex accounts for row insertion pushing down subsequent indices
        newIndex = index + rowIncreaseCounter

        # Check if newIndex is a multiple of frameSeq; enqures final df can be grouped by frameSeq
        if ((newIndex+1)%frameSeq != 0):
            # Get the last row in a class split group and duplicate it to complete the group
            rowCopy = df.iloc[newIndex]
            numCopies = frameSeq - ((newIndex+1)%frameSeq)
            rowCopies = pd.concat([rowCopy] * numCopies, axis=1).transpose()

            # Insert duplicated rows after newIndex in df
            df = pd.concat([df.iloc[:newIndex], rowCopies, df.iloc[newIndex:]]).reset_index(drop=True)
            
            rowIncreaseCounter += (frameSeq - ((newIndex+1)%frameSeq))
    return df



def flatten_input_spatial(df, frameSeq):
    """
    Flatten input for ML model w/ row-wise concatenation; demonstrates spatial change (how WHOLE frame changes over time)
    
    Parameters:
    df (DataFrame): dataframe to flatten
    frameSeq (number): length of desired frame sequence

    Return:
    featMatrix_flattened (numPy arr): flattened feature matrix
    classVector (numPy arr): vector of class labels that correspond to flattened feature matrix
    """
    
    # Seperate x and y values
    featMatrix = df.drop('label', axis=1)
    classVector = df['label']

    # Get groups of feature matrix rows for flattening
    featMatrix_groups = featMatrix.groupby(np.arange(len(featMatrix)) // frameSeq)

    # Apply a function to flatten each group into a 1D NumPy array
    featMatrix_flattened = featMatrix_groups.apply(lambda x: x.values.ravel()).reset_index(drop=True)

    # Get every frameSeq-th class
    classVector = classVector.iloc[::frameSeq].reset_index(drop=True)
    
    # Convert pandas series to numpy arrays
    featMatrix_flattened = np.array(featMatrix_flattened.tolist())
    classVector = classVector.to_numpy()

    # Write featMatrix to CSV, if desired
    # featMatrix_flattened.tofile("./SVM/flat_spatial.csv", sep=',')

    #return flattened feat matrix and class vector
    return featMatrix_flattened, classVector



def flatten_input_temporal(df, frameSeq):
    """
    Flatten input for ML model w/ column-wise concatenation; demonstrates temporal change (how EACH COORDINATE changes over time)
    
    Parameters:
    df (DataFrame): dataframe to flatten
    frameSeq (number): length of desired frame sequence

    Return:
    featMatrix_flattened (numPy arr): flattened feature matrix
    classVector (numPy arr): vector of class labels that correspond to flattened feature matrix
    """

    featMatrix = df.drop('label', axis=1)
    classVector = df['label']

    # Reshape the DataFrame by grouping rows into chunks of frameSeq
    featMatrix_groups = featMatrix.groupby(np.arange(len(featMatrix)) // frameSeq)

    # Apply a function to flatten each group column-wise
    featMatrix_flattened = featMatrix_groups.apply(lambda x: x.values.flatten('F'))

    # Reset index to create a flattened DataFrame
    featMatrix_flattened = featMatrix_flattened.reset_index(drop=True)

    # Get every frameSeq-th class
    classVector = classVector.iloc[::frameSeq].reset_index(drop=True)

    # Convert pandas series to numpy arrays
    featMatrix_flattened = np.array(featMatrix_flattened.tolist())
    classVector = classVector.to_numpy()

    # Write featMatrix to CSV, if desired
    # featMatrix_flattened.tofile("./SVM/flat_temporal.csv", sep=',')

    # Return flattened feat matrix and class vector
    return featMatrix_flattened, classVector



def modify_input_data(df, frameSeq, flatteningType):
    """
    Expand a dataframe, flatten it, and split it into a feature matrix and class label vector;
        wrapper function for insert_rows() and flattening functions (flatten_input_spatial() and flatten_input_temporal())
    
    Parameters:
    df (DataFrame): dataframe to preprocess
    frameSeq (number): length of frame sequence
    flatteningType (str): specify spatial ("s") or temporal ("t") flattening

    Return: 
    Return vars are from flatten_input_spatial() or flatten_input_temporal():
        featMatrix (numPy arr): feature matrix after df has been expanded, flattened, and split into features and class labels
        classVector (numPy): class vector after df has been expanded, flattened, and split into features and class labels
    """

    # EXPAND ROWS
    # Get indices where the class labels change
    #   diff() returns T/F list where the class label changes
    classSplits = df.iloc[:, df.shape[1]-1].diff().ne(0)
    classSplits = np.where(classSplits.iloc[1:])[0]

    # Include the index of the last item in df in case the expanded array is not a multiple of frameSeq;
    #   ensures last group is still the length of frameSeq
    classSplits = np.append(classSplits, df.shape[0] - 1)
    
    # Insert copies of frames based on class splits
    df = insert_rows(df, classSplits, frameSeq)


    # FLATTEN ROWS
    # Call appropriate flattening function based on specified flatteningType
    if(flatteningType == 's'):
        return flatten_input_spatial(df, frameSeq)
    elif(flatteningType == 't'):
        return flatten_input_temporal(df, frameSeq)
    else:
        print("Error, nonvalid flattening type specified")
        return None



# MACHINE LEARNING MODEL: SVM
def find_best_SVM(x_train, y_train, x_test, y_test, seed, nameExt):
    """
    Conduct a grid search to find the best combination of parameters to yield the highest testing accuracy for the SVM

    Parameters:
    x_train (numPy arr): training features
    y_train (numPy arr): training classes
    x_test (numPy arr): testing features
    y_test (numPy arr): testing classes
    seed (number): random seed for reproducibility
    nameExt (str): extension to add for file name to print best parameter combinations

    Return:
    None
    Print best parameters and highest accuracy and write combinations to text file.
    """

    # Set possible hyperparameters
    c = [1, 2, 10, 100]
    degree = [1, 2, 3]
    kernel = ['linear', 'poly', 'rbf']
    decision_function_shape = ['ovo', 'ovr']

    best_accuracy = 0
    best_parameters = None

    # Check ParameterOptions folder exists. If not, create it
    if not os.path.exists("./SVM/ParameterOptions/"):
        os.makedirs("./SVM/ParameterOptions/")

    fileName = './SVM/ParameterOptions/best_svm_parameters' + nameExt + '.txt'

    with open(fileName, 'w') as file:
        file.write("")
    for c_val in c:
        for d_val in degree:
            for k_val in kernel:
                for dfs_val in decision_function_shape:
                    # Initialize svm classifier
                    clf = svm.SVC(C=c_val, degree=d_val, kernel=k_val, decision_function_shape=dfs_val, random_state=seed)

                    # Fit SVM to training data
                    clf.fit(x_train, y_train)

                    # Predict classes with SVM
                    y_pred = clf.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    # >= instead of = means you will print the LAST set of parameters with the highest accuracy (since some sets may be tied for highest acc)
                    # > means you will keep the first set with the highest accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameters = ("Highest SVM accuracy so far: " + str(best_accuracy) + "\n"
                                        + "Parameters: c=" + str(c_val) + ", degree=" + str(d_val)
                                        + ", kernel=" + k_val + ", decision function shape=" + dfs_val + "\n")
                    # Write sets of best parameters (multiple groups might be tied for the same highest accuracy)
                        with open(fileName, 'a') as file:
                            file.write(best_parameters)
    # Print best parameters outside the loop
    print(best_parameters)



# Will need to figure out best output (i.e. list of classifications, testing acc, etc.)
    # + best way to save SVM so it can keep training as the app is used more
def create_svm(x_train, y_train, x_test, y_test, svc_c, svc_degree, svc_kernel, svc_shape, seed, nameExt):
    """
    Based on results of find_best_SVM(), use the best parameters to create an SVM

    Parameters:
    x_train (numPy arr): training features
    y_train (numPy arr): training classes
    x_test (numPy arr): testing features
    y_test (numPy arr): testing classes
    svc_c (number): C of SVM
    svc_degree (number): degree of SVM
    svc_kernel (str): kernel function of SVM
    svc_shape (str): decision function shape of SVM
    seed (number): random seed for reproducibility
    nameExt (str): keyword for file name to write SVM results

    Print and write SVM parameters and training/testing accuracies to a file.
    """

    # Create and train svm model
    clf = SVC(C=svc_c, degree=svc_degree, kernel=svc_kernel, decision_function_shape=svc_shape, random_state=seed)
    clf.fit(x_train, y_train)

    # Predictions and accuracy evaluation
    train_predictions = clf.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    # Test predictions and evaluation
    test_predictions = clf.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Check Results folder exists. If not, create it
    if not os.path.exists("./SVM/Results/"):
        os.makedirs("./SVM/Results/")
        
    fileName="./SVM/Results/svm_"+nameExt+".txt"

    # Write SVM parameters and accuracies to file
    with open(fileName, 'w') as file:
        file.write("Parameters: c="+ str(svc_c) + ", degree=" + str(svc_degree)
                    + ", kernel=" + svc_kernel + ", decision function shape=" + svc_shape + "\n")
        file.write(f'Training accuracy: {train_accuracy:.2f}\n')
        file.write(f'Testing accuracy: {test_accuracy:.2f}')


    # Print SVM parameters and accuracies
    print("Parameters: c="+ str(svc_c) + ", degree=" + str(svc_degree)
                    + ", kernel=" + svc_kernel + ", decision function shape=" + svc_shape)
    print(f'Training accuracy: {train_accuracy:.2f}')
    print(f'Testing accuracy: {test_accuracy:.2f}')
