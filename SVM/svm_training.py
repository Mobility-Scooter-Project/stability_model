from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from joblib import dump

# LATER: normalize coordinates
#   for each value, find the smallest coordinates and normalize them as 0
#   for other coordinates, subtract from smallest coor
# may need to balance classes w/ augmentation

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
    fullDF = fullDF[fullDF.label != 'UnlabeledEND OF DATA COLLECTION']

    # Convert class labels to binary w/ label mapping
    encoding = {'Stable': 1, 'Minimum Sway': 0, 'Sway rq UE sup': 0}
    fullDF['label'] = fullDF['label'].replace(encoding)
    fullDF = fullDF.reset_index(drop=True)

    return fullDF


def split_data(df, testingRatio):
    """
    Split training/testing data based on split ratio.

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



# TRAINING DATA PREPROCESSING
def insert_rows(df, listOfIndices, frameSeq):
    """
    Expand a dataframe to become a multiple of <frameSeq> by inserting rows.
        Last row in a group with the same class (an index in listOfIndices) will be copied enough times to complete the group of <frameSeq>.

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



def flatten_input_spatial(featMatrix, frameSeq):
    """
    Flatten feature matrix for ML model w/ row-wise concatenation; 
        demonstrates spatial change (how WHOLE frame changes over time).
    
    Parameters:
    featMatrix (DataFrame): dataframe to flatten
    frameSeq (number): length of desired frame sequence

    Return:
    featMatrix_flattened (numPy arr): flattened feature matrix
    """

    # Get groups of feature matrix rows for flattening
    featMatrix_groups = featMatrix.groupby(np.arange(len(featMatrix)) // frameSeq)

    # Apply a function to flatten each group into a 1D NumPy array
    featMatrix_flattened = featMatrix_groups.apply(lambda x: x.values.ravel()).reset_index(drop=True)

    # Convert pandas series to numpy arrays
    featMatrix_flattened = np.array(featMatrix_flattened.tolist())

    # Write featMatrix to CSV, if desired
    # featMatrix_flattened.tofile("./SVM/flat_spatial.csv", sep=',')

    #return flattened feat matrix and class vector
    return featMatrix_flattened


def flatten_input_temporal(featMatrix, frameSeq):
    """
    Flatten input for ML model w/ column-wise concatenation; 
        demonstrates temporal change (how EACH COORDINATE changes over time).
    
    Parameters:
    featMatrix (DataFrame): dataframe to flatten
    frameSeq (number): length of desired frame sequence

    Return:
    featMatrix_flattened (numPy arr): flattened feature matrix
    """

    # Reshape the DataFrame by grouping rows into chunks of frameSeq
    featMatrix_groups = featMatrix.groupby(np.arange(len(featMatrix)) // frameSeq)

    # Apply a function to flatten each group column-wise
    featMatrix_flattened = featMatrix_groups.apply(lambda x: x.values.flatten('F'))

    # Reset index to create a flattened DataFrame
    featMatrix_flattened = featMatrix_flattened.reset_index(drop=True)

    # Convert pandas series to numpy arrays
    featMatrix_flattened = np.array(featMatrix_flattened.tolist())

    # Write featMatrix to CSV, if desired
    # featMatrix_flattened.tofile("./flat_temporal.csv", sep=',')

    # Return flattened feat matrix and class vector
    return featMatrix_flattened


def flatten_training_classes(classVector, frameSeq):
    """
    Flatten the class vector for the training phase.

    Parameters:
    classVector (DataFrame): class vector to flatten
    frameSeq (number): number of frames in each group to classify
    """
    # Get every frameSeq-th class
    classVector = classVector.iloc[::frameSeq].reset_index(drop=True)
    
    # Convert pandas series to numpy array
    classVector = classVector.to_numpy()

    return classVector


def preprocess_training_data(df, frameSeq, flatteningType):
    """
    Expand a dataframe, flatten it, and split it into a feature matrix and a class label vector;
        wrapper function for insert_rows() and flattening functions (flatten_input_spatial(), flatten_input_temporal(), and flatten_training_classes()).
    
    Parameters:
    df (DataFrame): dataframe to preprocess
    frameSeq (number): length of frame sequence
    flatteningType (str): specify spatial ("s") or temporal ("t") flattening

    Return: 
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

    # Split feature matrix and classVector in df
    featMatrix = df.drop('label', axis=1)
    classVector = df['label']
    
    # FLATTEN ROWS
    # Call appropriate flattening function based on specified flatteningType
    if(flatteningType == 's'):
        featMatrix = flatten_input_spatial(featMatrix, frameSeq)
    elif(flatteningType == 't'):
        featMatrix = flatten_input_temporal(featMatrix, frameSeq)
    else:
        print("Error, nonvalid flattening type specified")
        return None
    
    # Get classes for flattened feature matrix
    classVector = flatten_training_classes(classVector, frameSeq)

    return featMatrix, classVector



# TESTING DATA PREPROCESSING
def round_df(df, frameSeq):
    """
    "Round" a dataframe down to become a multiple of <frameSeq> by deleting the last incomplete group, if any

    Parameters:
    df (DataFrame): dataframe to round
    frameSeq (number): length of desired frame sequence (i.e. 64 frames)

    Return:
    df (DataFrame): "Rounded" df; same as passed df parameter
    """

    # If the dataframe is not a multiple of frameSeq, just delete the last incomplete group
    if (len(df)%frameSeq != 0):
        rowsToDelete = len(df)%frameSeq
        df = df.drop(df.tail(rowsToDelete).index)

    return df


def flatten_testing_classes(classVector, frameSeq):
    """
    Flatten the class vector for the testing phase.

    Parameters:
    classVector (DataFrame): class vector to flatten
    frameSeq (number): number of frames in each group to classify
    """

    # Separate class vector into groups of <frameSeq>, then find majority label to use as the true class for a group
    return classVector.groupby(classVector.index // frameSeq).apply(get_majority_label)

def get_majority_label(group):
    """
    Helper function for flatten_testing_classes().
        Return the majority class of a given group.

    Parameters:
    group (DataFrame): the group from class vector to classify.

    Return:
    number: majority class label (encoded)
    """
    # Count occurence of each label in group
    counts = group.value_counts()

    # Return label with highest count
    return counts.idxmax()


def preprocess_testing_data(df, frameSeq, flatteningType):
    """
    "Round" down a dataframe, flatten it, and split it into a feature matrix and a class label vector;
        wrapper function for round_df() and flattening functions (flatten_input_spatial(), flatten_input_temporal(), and flatten_testing_classes()).
    
    Parameters:
    df (DataFrame): dataframe to preprocess
    frameSeq (number): length of frame sequence
    flatteningType (str): specify spatial ("s") or temporal ("t") flattening

    Return: 
    featMatrix (numPy arr): feature matrix after df has been rounded down, flattened, and split into features and class labels
    classVector (numPy): class vector after df has been rounded down, flattened, and split into features and class labels
    """ 

    # Round input down to nearest multiple of frameSeq
    df = round_df(df, frameSeq)

    # Split feature matrix and true classes 
    featMatrix = df.iloc[:, :-1]
    featMatrix = featMatrix.reset_index(drop=True)
    classVector = df.iloc[:, -1]
    classVector = classVector.reset_index(drop=True)
    
    # Flatten feature matrix
    if(flatteningType == 's'):
        featMatrix = flatten_input_spatial(featMatrix, frameSeq)
    elif(flatteningType == 't'):
        featMatrix = flatten_input_temporal(featMatrix, frameSeq)
    else:
        print("Error, nonvalid flattening type specified")
        return None
    
    # Flatten class vector
    classVector = flatten_testing_classes(classVector, frameSeq)

    return featMatrix, classVector




# MACHINE LEARNING MODEL: SVM
def find_best_SVM(x_train, y_train, x_test, y_test, seed):
    """
    Conduct a grid search to find the best combination of parameters to yield the highest testing accuracy for the SVM.

    Parameters:
    x_train (numPy arr): training features
    y_train (numPy arr): training classes
    x_test (numPy arr): testing features
    y_test (numPy arr): testing classes
    seed (number): random seed for reproducibility

    Return:
    None
    Print best parameters and highest accuracy.
    """

    # Set possible hyperparameters
    c = [1, 2, 10, 100]
    degree = [1, 2, 3]
    kernel = ['linear', 'poly', 'rbf']
    decision_function_shape = ['ovo', 'ovr']

    best_accuracy = 0
    best_parameters_str = None
    best_c = 0
    best_degree = 0
    best_kernel = None
    best_shape = None

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
                        best_c = c_val
                        best_degree = d_val
                        best_kernel = k_val
                        best_shape = dfs_val
                        best_parameters_str = ("Highest SVM testing accuracy so far: " + str(best_accuracy) + "\n"
                                        + "Parameters: c=" + str(c_val) + ", degree=" + str(d_val)
                                        + ", kernel=" + k_val + ", decision function shape=" + dfs_val + "\n")
    # Print best parameters outside the loop
    print(best_parameters_str)
    
    return best_c, best_degree, best_kernel, best_shape



def create_svm(x_train, y_train, x_test, y_test, svc_c, svc_degree, svc_kernel, svc_shape, seed, frameSeq, samplingRate, nameExt, jobFile):
    """
    Based on results of find_best_SVM(), use the best parameters to create an SVM.

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

    Return:
    None
    Print and write SVM parameters and training/testing accuracies to a file.
    Save SVM to a job file.
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
    if not os.path.exists("./Results/"):
        os.makedirs("./Results/")
        
    fileName="./Results/svm"+nameExt+".txt"

    # Write SVM parameters and accuracies to file
    with open(fileName, 'w') as file:
        file.write("Frame sequence length: " + str(frameSeq) + ", sampling rate: 1 in " + str(samplingRate) + "\n")
        file.write("Parameters: c="+ str(svc_c) + ", degree=" + str(svc_degree)
                    + ", kernel=" + svc_kernel + ", decision function shape=" + svc_shape + "\n")
        file.write(f'Training accuracy: {train_accuracy:.6f}\n')
        file.write(f'Testing accuracy: {test_accuracy:.6f}')


    # Print SVM parameters and accuracies
    print("Frame sequence length: ", frameSeq, ", sampling rate: 1 in", samplingRate)
    print("Parameters: c="+ str(svc_c) + ", degree=" + str(svc_degree)
                    + ", kernel=" + svc_kernel + ", decision function shape=" + svc_shape)
    print(f'Training accuracy: {train_accuracy:.6f}')
    print(f'Testing accuracy: {test_accuracy:.6f} \n\n')

    # Save SVM model for testing
    dump(clf, jobFile)
    


def save_params(frameSeq, samplingRate, fileName):
    """
    Save the frame sequence length and sampling rate values to a text file.
    
    Parameters: 
    frameSeq (number): length of desired frame sequence
    samplingRate (number): rate of samples to collect (i.e. if samplingRate = 5, take every 5th row)
    
    Return:
    None
    Write values to a text file.
    """
    with open(fileName, 'w') as file:
        file.write(str(frameSeq)+",")
        file.write(str(samplingRate))




def main():
    '''
    Main function to run the SVM.
        Read data, split for training/testing, expand/round, and flatten spatially (row-wise) and temporally (column-wise)
        for input for temporal SVM. Spatial SVM was removed since its results were redundant.
    '''

    dataDir = "./" # directory with data for SVM
    fileKeyword = "Front" # keyword in desired data file names
    samplingRate = 4 # rate to sample data
    samplingFormula = lambda x: x % samplingRate != 0
    frameSeq = 20 # frame sequence length to group data for SVM input
    testingRatio = 0.3 # testing/training data split
    save_params(frameSeq, samplingRate, "./svm_parameters.txt")

    # Read all data from CSV's in specified directory
    fullDF = read_data(dataDir, samplingFormula, fileKeyword)

    # Split training/testing data by 70:30
    trainingDF, testingDF = split_data(fullDF, testingRatio)

    # Temporal SVM
    x_train_t, y_train_t = preprocess_training_data(trainingDF, frameSeq, "t")
    x_test_t, y_test_t = preprocess_testing_data(testingDF, frameSeq, "t")
    
    # Find best parameters for SVM
    best_c, best_degree, best_kernel, best_shape = find_best_SVM(x_train_t, y_train_t, x_test_t, y_test_t, seed=10)
    
    # Use best parameters to create temporal SVM
    nameExtension = "_temporal_"+str(frameSeq)+"_"+str(samplingRate)
    create_svm(x_train_t, y_train_t, x_test_t, y_test_t, svc_c=best_c, svc_degree=best_degree, svc_kernel=best_kernel, svc_shape=best_shape, seed=10, frameSeq=frameSeq, samplingRate=samplingRate, nameExt=nameExtension, jobFile="./svm.joblib")
    

if __name__ == "__main__":
    main()