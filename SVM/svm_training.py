from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score, average_precision_score, f1_score, make_scorer
from sklearn.utils import resample
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import os
from joblib import dump
import csv
import sys


# PREPROCESSING
def read_data(dir, samplingRate, keyword):
    """
    Return a dataframe with all relevant CSV's concatenated together.
    
    Parameters:
    dir (str): name of directory with all the CSV's
    samplingRate (number): get every nth record in a CSV to reduce number of samples
    keyword (str): find CSV files with the keyword in their names

    Return:
    fullDF (DataFrame): dataframe with all of the relevant data for the machine learning model
    """

    fullDF = pd.DataFrame()
    # Iterate through all front view CSV's and concatenate into 1 giant DF
    for folder, subfolder, files in os.walk(dir):
        for fileName in files:
            if keyword in fileName:
                filePath = os.path.join(folder, fileName)
                tempDF = pd.read_csv(filePath, skiprows=samplingRate)
                fullDF = pd.concat([fullDF, tempDF], axis=0, ignore_index=True)
    
    # Remove any rows with label "Unlabeled"
    fullDF = fullDF[fullDF.label != 'Unlabeled']
    fullDF = fullDF[fullDF.label != 'UnlabeledEND OF DATA COLLECTION']

    # Convert class labels to binary w/ label mapping
    encoding = {'Stable': 0, 'Minimum Sway': 1, 'Sway rq UE sup': 1}
    fullDF['label'] = fullDF['label'].replace(encoding)
    fullDF = fullDF.reset_index(drop=True)

    # Return dataframe of all data
    return fullDF



def split_data(df, testingRatio, frameSeq): 
    """
    Expand data and split training/testing data based on split ratio.

    Parameters:
    df (DataFrame): dataframe to split
    testingRatio (number): the fraction of data to set aside for testing during training/testing split
    frameSeq (number): desired length of frame sequence groups (i.e. 30 frames in a group to be flattened together)

    Return:
    trainingDF (DataFrame): dataframe w/ training data
    testingDF (DataFrame): dataframe w/ testing data
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


    # GROUP ROWS BY CLASS LABEL
    stable_rows = df[df['label'] == 0].reset_index(drop=True)
    unstable_rows = df[df['label'] == 1].reset_index(drop=True)

    # Split df into groups of <frameSeq>
    numGroups_stable = len(stable_rows) // frameSeq
    stable_groups = [stable_rows.iloc[i*frameSeq:(i+1)*frameSeq] for i in range(numGroups_stable)]

    numGroups_unstable = len(unstable_rows) // frameSeq
    unstable_groups = [unstable_rows.iloc[i*frameSeq:(i+1)*frameSeq] for i in range(numGroups_unstable)]

    # Shuffle groups of each class
    np.random.shuffle(stable_groups)
    np.random.shuffle(unstable_groups)

    # Get number of stable and unstable groups for training (and testing)
    trainingRatio = 1 - testingRatio
    num_training_stable_groups = int(trainingRatio * numGroups_stable)
    num_training_unstable_groups = int(trainingRatio * numGroups_unstable)

    # Split the groups for training/testing
    training_stable_groups = stable_groups[:num_training_stable_groups]
    testing_stable_groups = stable_groups[num_training_stable_groups:]

    training_unstable_groups = unstable_groups[:num_training_unstable_groups]
    testing_unstable_groups = unstable_groups[num_training_unstable_groups:]

    # Concat groups to get final training and testing df's
    tempDF = pd.concat(training_stable_groups, ignore_index=True)
    tempDF_2 = pd.concat(training_unstable_groups, ignore_index=True)
    trainingDF = pd.concat([tempDF, tempDF_2], ignore_index=True)

    tempDF = pd.concat(testing_stable_groups, ignore_index=True)
    tempDF_2 = pd.concat(testing_unstable_groups, ignore_index=True)
    testingDF = pd.concat([tempDF, tempDF_2], ignore_index=True)

    # Can export training and testing df's, if desired
    # trainingDF.to_csv("training.csv", index=False)
    # testingDF.to_csv("testing.csv", index=False)

    return trainingDF, testingDF



def insert_rows(df, listOfIndices, frameSeq):
    """
    Expand a dataframe to become a multiple of <frameSeq> by inserting rows.
        Last row in a group with the same class (an index in listOfIndices) will be copied enough times to complete the group of <frameSeq>.

    Parameters:
    df (DataFrame): dataframe to expand
    listOfIndices (list): list of indices where class label changes; an index is the LAST row in a group with the same class label
    frameSeq (number): number of frames in a sequence (flattened together to make 1 sample for the SVM)

    Return:
    df (DataFrame): Expanded df; same as passed df parameter
    """
    
    # Counter to track how row insertion affects all subsequent indices
    rowIncreaseCounter = 0
    
    for index in listOfIndices:
        # newIndex accounts for row insertion pushing down subsequent indices
        newIndex = index + rowIncreaseCounter

        # Check if newIndex is a multiple of frameSeq; ensures final df can be grouped by frameSeq
        if ((newIndex+1)%frameSeq != 0):
            # Get the last row in a class split group and duplicate it to complete the group
            rowCopy = df.iloc[newIndex]
            numCopies = frameSeq - ((newIndex+1)%frameSeq)
            rowCopies = pd.concat([rowCopy] * numCopies, axis=1).transpose()

            # Insert duplicated rows after newIndex in df
            df = pd.concat([df.iloc[:newIndex], rowCopies, df.iloc[newIndex:]]).reset_index(drop=True)
            
            rowIncreaseCounter += (frameSeq - ((newIndex+1)%frameSeq))
    
    return df



def flatten_features(featMatrix, frameSeq):
    """
    Flatten input for ML model w/ column-wise concatenation; 
        demonstrates temporal change (how EACH COORDINATE changes over time).
    *Row-wise concatenation (spatial change/how WHOLE frame changes over time) produces exact same results.
    
    Parameters:
    featMatrix (DataFrame): dataframe to flatten
    frameSeq (number): number of frames in a sequence (flattened together to make 1 sample for the SVM)

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

    # Return flattened feat matrix
    return featMatrix_flattened



def flatten_classes(classVector, frameSeq):
    """
    Flatten the class vector.

    Parameters:
    classVector (DataFrame): class vector to flatten
    frameSeq (number): number of frames in a sequence (flattened together to make 1 sample for the SVM)
    
    Return:
    classVector (DataFrame): flattened class vector (same input variable but changed)
    """

    # Get every frameSeq-th class
    classVector = classVector.iloc[::frameSeq].reset_index(drop=True)
    
    # Convert pandas series to numpy array
    classVector = classVector.to_numpy()

    return classVector



def preprocess_data(df, frameSeq, resample):
    """
    Flatten a dataframe and split it into a feature matrix and a class label vector;
        wrapper function for flattening functions (flatten_features() and flatten_classes()).
    
    Parameters:
    df (DataFrame): dataframe to preprocess
    frameSeq (number): number of frames in a sequence (flattened together to make 1 sample for the SVM)
    resample (boolean): Flag to resample the data to balance the classes; ONLY for training.
        You CANNOT balance classes for testing.
        CANNOT set to True UNLESS you delete all class_weight='balanced' parameters in SVC (SVM) functions
            because resampling here and balancing class weight in SVMs may cause data distortion due to overcompensating for class imbalance
    
    Return: 
    featMatrix (numPy arr): feature matrix after df has been expanded, flattened, and split into features and class labels
    classVector (numPy): class vector after df has been expanded, flattened, and split into features and class labels
    class_distribution (array): class distributions of flattened data
    """

    # Split feature matrix and classVector in df
    featMatrix = df.drop('label', axis=1)
    classVector = df['label']
    

    # FLATTEN ROWS
    # Call appropriate flattening function based on specified flatteningType
    featMatrix = flatten_features(featMatrix, frameSeq)
    
    # Get classes for flattened feature matrix
    classVector = flatten_classes(classVector, frameSeq)

    # Can only be used if you delete class_weight='balanced' in SVC() calls (scattered throughout code)
    if(resample == True):
        featMatrix, classVector = resample_data(featMatrix, classVector)

    # Output class distribution, if desired
    temp = pd.DataFrame(classVector)
    class_distribution = calc_class_distribution(temp)

    # Return flattened feature matrix, flattened class vector, and class distributions
    return featMatrix, classVector, class_distribution



def calc_class_distribution(df):
    '''
    Calculate the stable and unstable class distributions of an input dataframe

    Parameters:
    df (DataFrame): input dataframe; should be a class vector

    Return:
    distribution (array): Array of class distributions as fractions
        i.e. [0.9, 0.1] for [stable, unstable]
    '''

    # Get class counts
    counts = df.value_counts()

    # Calculate class distribution as fractions
    stable_dist = round(counts.get(0,0) / len(df), 4)
    unstable_dist = round(counts.get(1,0) / len(df), 4)
    distribution = [stable_dist, unstable_dist]

    return distribution



def resample_data(featMatrix, classVector):
    '''
    Resample data to account for class imbalance
        Can only use if you delete class_weight='balanced' in ALL SVC() calls (scattered throughout code)
            and if resample=True in preprocess_data() calls
        Class imbalance is already accounted for with class_weight='balanced' parameter
        If you resample data AND balance class weights in SVM's, data could become distorted due to overcompensation
    
    Parameters:
    featMatrix (DataFrame): feature matrix
    classVector (DataFrame): vector of stable/unstable classes
    
    Return:
    featMatrix (DataFrame): newly resampled feature matrix
    classVector (DataFrame): newly resampled vector of stable/unstable classes
    '''

    # Set the desired minority and majority class ratios for the final output
    final_minority_ratio = 0.4
    final_majority_ratio = 1 - final_minority_ratio
    
    # Current counts
    n_minority = sum(classVector == 1)
    n_majority = sum(classVector == 0)
    total = len(classVector)

    # Desired final counts
    desired_n_minority = int(final_minority_ratio * total)
    desired_n_majority = int(final_majority_ratio * total)

    # Resampling
    # Can use other resampling techniques, such as SMOTE (oversampling), BorderlineSMOTE (oversampling), etc.
    # Oversample minority
    if n_minority < desired_n_minority:
        oversampler = ADASYN(sampling_strategy={1: desired_n_minority})
        featMatrix, classVector = oversampler.fit_resample(featMatrix, classVector)
    
    # Undersample majority
    if n_majority > desired_n_majority:       
        undersampler = RandomUnderSampler(sampling_strategy={0: desired_n_majority})
        featMatrix, classVector = undersampler.fit_resample(featMatrix, classVector)

    # Return the newly resampled data
    return featMatrix, classVector




# MACHINE LEARNING MODEL: SVM
def find_best_SVM(x_train, y_train, x_test, y_test, seed):
    '''
    Find the best SVM hyperparameters for the given data

    Parameters:
    x_train (numPy arr): training features
    y_train (numPy arr): training classes
    x_test (numPy arr): testing features
    y_test (numPy arr): testing classes
    seed (number): random seed for reproducibility

    Return:
    best_c (number): best SVM c value
    best_degree(number): best SVM degree value
    best_kernel(number): best SVM kernel
    best_shape (number): best SVM shape
    '''

    # Define the SVC model
    model = SVC(class_weight='balanced', random_state=seed)

    # Define the parameter grid
    param_grid = {
        'C': [1, 2, 10, 100],
        'degree': [1, 2, 3],
        'kernel': ['linear', 'poly', 'rbf'],
        'decision_function_shape': ['ovo', 'ovr']
    }

    # Define the scoring metric
    scorer = make_scorer(recall_score, average='binary')

    # Create a stratified k-fold cross-validator
    # If you have more computational power, n_splits=10 could produce better (less biased) results
    #   shuffle samples for better generalization
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Set up the grid search to choose hyperparameters based on highest recall score
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=cv)

    # Fit the grid search
    grid_search.fit(x_train, y_train)

    # Get the resulting best hyperparameters
    best_parameters = grid_search.best_params_
    best_c = best_parameters['C']
    best_degree = best_parameters['degree']
    best_kernel = best_parameters['kernel']
    best_shape = best_parameters['decision_function_shape']

    # Use the best model to classify the test set
    y_pred = grid_search.best_estimator_.predict(x_test)

    # Calculate the recall score on the test data
    test_recall = recall_score(y_test, y_pred, average='binary')

    # Output the best hyperparameters and testing recall for this SVM
    print("Best hyperparameters:", grid_search.best_params_)
    print("Testing recall score with best hyperparameters:", test_recall)

    # Return the best hyperparameters
    return best_c, best_degree, best_kernel, best_shape



def create_svm(x_train, y_train, x_test, y_test, svc_c, svc_degree, svc_kernel, svc_shape, seed, saveSVM, jobFileName):
    """
    Based on results of find_best_SVM(), use the best hyperparameters to create an SVM.
        Output anamoly metrics (accuracy, precision, recall, F1, MCC, AP)
        Save the SVM to a joblib, if desired

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
    saveSVM (boolean): flag to save the SVM that will be created in this function to a joblib file
    jobFileName (str): desired name for the output SVM joblib file

    Return:
    train_accuracy (number): Training accuracy
    test_accuracy (number): Testing accuracy
    mcc (number): Matthews Correlation Coefficient
    precision (number): precision score
    recall (number): recall score
    ap_score (number): average precision score
    f1 (number): F1 score
    (Optional) Save SVM to a job file.
    """

    # Create and train svm model
    clf = SVC(C=svc_c, degree=svc_degree, kernel=svc_kernel, decision_function_shape=svc_shape, random_state=seed, class_weight='balanced')
    clf.fit(x_train, y_train)

    # Predictions and accuracy evaluation
    train_predictions = clf.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    # Test predictions and evaluation
    test_predictions = clf.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Get conf_matrix and print, if desired
    # conf_matrix = confusion_matrix(y_test, test_predictions)
    # print(conf_matrix)
    # TP = conf_matrix[1, 1]
    # TN = conf_matrix[0, 0]
    # FP = conf_matrix[0, 1]
    # FN = conf_matrix[1, 0]

    # # Print confusion matrix
    # print("True Positives:", TP)
    # print("True Negatives:", TN)
    # print("False Positives:", FP)
    # print("False Negatives:", FN)

    # Calculate Matthews Correlation Coefficient (measures how well SVM makes predictions)
    mcc = matthews_corrcoef(y_test, test_predictions)

    # Calculate precision, recall, AP score, and F1 score
    precision = precision_score(y_test, test_predictions)
    recall = recall_score(y_test, test_predictions)
    ap_score = average_precision_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions)
    
    # Save SVM model for testing
    if(saveSVM == True):
        dump(clf, jobFileName)

    # Return metrics
    return train_accuracy, test_accuracy, mcc, precision, recall, ap_score, f1
    


def save_params(frameSeq, samplingRate, fileName):
    """
    Save the frame sequence length and sampling rate values to a text file
        (During testing) SVM joblib file needs to know frame sequence length and sampling rate to work
    
    Parameters: 
    frameSeq (number): number of frames in a sequence (flattened together to make 1 sample for the SVM)
    samplingRate (number): rate of samples to collect (i.e. if samplingRate = 5, take every 5th row)
    fileName (str): desired name for output file

    Return:
    None
    Write values to a text file.
    """
    with open(fileName, 'w') as file:
        file.write(str(frameSeq)+",")
        file.write(str(samplingRate))



def find_best_parameters(dataDir, fileKeyword, testingRatio, iterations, gridSearch):
    '''
    Find the best hyperparameters and frameSequence-samplingRate combo for our SVM (based on recall scores)

    Find the best hyperparameters for each frame sequence-sampling rate (fS-sR) combo by performing a grid search in find_best_SVM();
        calculating the average training/testing accuracies, precision, recall, F1, MCC, and AP;
        and choosing the best fS-sR combo and hyperparameters based on the highest average recall amongst the <iterations>
        
    Parameters:
    dataDir (str): directory with data for SVM
    fileKeyword (str): keyword to match in file names to pull relevant data
    testingRatio (number): the fraction of data to set aside for testing during training/testing split
        I.e. 70/30 split means testingRatio = 0.3
    iterations (number): number of times to test each frame sequence-sampling rate combination to calculate average metrics
    gridSearch (boolean): If true, run grid search in find_best_SVM(). 
        If false, use hyperparameters.csv (WHICH MUST EXIST IF gridSearch=False)

    Return:
    best_c (number): best SVM c value
    best_degree(number): best SVM degree value
    best_kernel(number): best SVM kernel
    best_shape (number): best SVM shape
    best_frameSeq (number): best frame sequence length
    best_samplingRate (number): best sampling rate of data
    
    Write anamoly metrics and (if not already existing) best SVM hyperparameters to files
    '''

    # List of possible frameSeq-samplingRate combos
    #   [frameSeq, samplingRate]
    parameters = [
        [20, 3], [20, 4], [20, 5],
        [30, 2], [30, 3], [30, 4],
        [40, 2], [40, 3],
        [50, 2],
        [60, 2]
    ]

    # DF to track best hyperparameters per frameSeq-samplingRate combo
    best_hyperparameters = pd.DataFrame()

    # Default hyperparameters w/ dummy values (will be overwritten)
    c = 0
    degree = 0
    kernel = None
    shape = None
    
    # Track highest anamoly metrics that come from testing each frameSeq-samplingRate combo
    highestAccuracyOverall = 0 # testing accuracy because testing accuracy is less important
    highestPrecisionOverall = 0
    highestRecallOverall = 0
    highestAPOverall = 0
    highestF1Overall = 0
    highestMCCOverall = 0


    # If gridSearch is true, create a file to keep track of best hyperparameters per frameSeq-samplingRate combo
    if(gridSearch == True):
        hyperparameter_headers = ["frameSeq", "samplingRate", "c", "degree", "kernel", "decision_func_shape"]
        with open("best_hyperparameters.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(hyperparameter_headers)

    # Create file to export anamoly metrics
    anamoly_metrics_headers = ["frameSeq", "samplingRate", "Training Accuracy", "Testing Accuracy", "Precision", "Recall",  "F1 Score", "AP Score", "MCC"]
    with open("anamoly_metrics.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(anamoly_metrics_headers)    


    # Iterate through each combo of frameSequence and samplingRate in parameters[] array
    for index, paramSet in enumerate(parameters):
        # Track the anamoly metrics
        totalTrainingScores = 0
        avgTrainingAccuracy = 0

        totalTestingScores = 0
        avgTestingAccuracy = 0

        totalMCCScore = 0
        avgMCC = 0

        totalPrecisionScores = 0
        avgPrecision = 0

        totalRecallScores = 0
        avgRecall = 0

        totalAPScores = 0
        avgAP = 0

        totalF1Scores = 0
        avgF1 = 0

        # Get sampling rate and frame sequence length for this loop
        samplingRate = paramSet[1] # rate to sample data
        samplingFormula = lambda x: x % samplingRate != 0
        frameSeq = paramSet[0] # frame sequence length to group data for SVM input
        print("\nframeSeq: "+str(frameSeq) + ", samplingRate: "+str(samplingRate))

        # If gridSearch if false, read the best hyperparameters from a file
        #   hyperparameters will be used for all 10 iterations
        if gridSearch == False:
            try:
                best_hyperparameters = pd.read_csv("best_hyperparameters.csv")
                print(best_hyperparameters)
                c, degree, kernel, shape = best_hyperparameters.loc[index, ["c", "degree", "kernel", "decision_func_shape"]]
                
            except FileNotFoundError:
                print("The best_hyperparameters.csv file could not be found. Please make sure it is the same directory or run gridSearch=True to generate the file.")
                sys.exit(1)
            
            # Handle exception where best_hyperparameters.csv file is empty or formatted incorrectly
            except Exception as e:
                print("Something went wrong while reading the best_hyperparameters.csv file. \n" \
                    "Please ensure file is not empty and has the following headers (in order): \n" \
                    "frameSeq, samplingRate, c, degree, kernel, decision_func_shape")
                sys.exit(1)

        
        # Read all data from CSV's in specified directory
        fullDF = read_data(dataDir, samplingFormula, fileKeyword)

        # Train SVM <iterations> times with different training/testing data splits
        for i in range(iterations):
            # Split training/testing data by 70:30
            trainingDF, testingDF = split_data(fullDF, testingRatio, frameSeq)

            # Preprocess training and testing data; get class distributions AFTER flattening
            x_train, y_train, training_distribution = preprocess_data(trainingDF, frameSeq, False)
            x_test, y_test, testing_distribution = preprocess_data(testingDF, frameSeq, False)

            # Grid search is most efficient and effective when performed just ONCE per samplingRate-frameSeq combo (i.e. the first iteration)
            if(gridSearch == True) and (i == 0):
                c, degree, kernel, shape = find_best_SVM(x_train, y_train, x_test, y_test, seed=10)
                
                # Write c, degree, kernel, and shape to file
                with open("best_hyperparameters.csv", 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frameSeq, samplingRate, c, degree, kernel, shape])
            
            # Retrieve the anamoly metrics after using the best hyperparameters to create an SVM
            trainingAcc, currAccuracy, mcc, precision, recall, ap_score, f1 = create_svm(x_train, y_train, x_test, y_test, svc_c=c, svc_degree=degree, svc_kernel=kernel, svc_shape=shape, seed=10, saveSVM=False, jobFileName=None)

            # Update avg accuracies
            totalTrainingScores += trainingAcc
            avgTrainingAccuracy = totalTrainingScores / (i+1)

            totalTestingScores += currAccuracy
            avgTestingAccuracy = totalTestingScores / (i+1)

            # Update avg anamoly metrics
            totalPrecisionScores += precision
            avgPrecision = totalPrecisionScores / (i+1)

            totalRecallScores += recall
            avgRecall = totalRecallScores / (i+1)

            totalF1Scores += f1
            avgF1 = totalF1Scores / (i+1)

            totalAPScores += ap_score
            avgAP = totalAPScores / (i+1)

            totalMCCScore += mcc
            avgMCC = totalMCCScore / (i+1)


        # Compare the avg stats of this frameSeq-samplingRate combo to EVERY other combo
        if(avgPrecision >= highestPrecisionOverall):
            highestPrecisionOverall = avgPrecision
        print("Avg precision:", avgPrecision)
        print("Highest precision", highestPrecisionOverall)

        # Update best OVERALL frame parameters and SVM hyperparameters based on recall score of this frameSeq-samplingRate combo and hyperparams
        if(avgRecall >= highestRecallOverall):
            highestRecallOverall = avgRecall
            best_c = c
            best_degree = degree
            best_kernel = kernel
            best_shape = shape
            best_frameSeq = frameSeq
            best_samplingRate = samplingRate 
        print("Avg recall:", avgRecall)
        print("Highest recall", highestRecallOverall)

        # Compare average F1 score of current combo to highest overall
        if(avgF1 >= highestF1Overall):
            highestF1Overall = avgF1
        print("Avg F1:", avgF1)
        print("Highest F1", highestF1Overall)

        # Compare average AP (area under the precision-recall curve (AUC-PR)) score of current combo to highest overall
        if(avgAP >= highestAPOverall):
            highestAPOverall = avgAP
        print("Avg AP:", avgAP)
        print("Highest AP", highestAPOverall)

        # Compare average MCC score of current combo to highest overall
        if(avgMCC >= highestMCCOverall):
            highestMCCOverall = avgMCC
        print("Avg MCC:", avgMCC)
        print("Highest MCC overall", highestMCCOverall)

        # Compare average testing accuracy of current combo to highest overall; 
        #   training accuracy does not matter as much for printing
        if(avgTestingAccuracy >= highestAccuracyOverall):
            highestAccuracyOverall = avgTestingAccuracy
        print("Avg testing accuracy: ", avgTestingAccuracy)
        print("Highest accuracy overall: ", highestAccuracyOverall)


        # Save frameSeq, samplingRate, avg accuracies, MCC, precision, recall, F1, and AP to a file
        with open("anamoly_metrics.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frameSeq, samplingRate, avgTrainingAccuracy, avgTestingAccuracy, avgPrecision, avgRecall, avgF1, avgAP , avgMCC])

    # return best frame params and hyperparams
    return best_c, best_degree, best_kernel, best_shape, best_frameSeq, best_samplingRate



def train_final_SVM(dataDir, fileKeyword, testingRatio, c, degree, kernel, shape, frameSeq, samplingRate, jobFileName):
    '''
    Train final SVM given desired hyperparameters, frame sequence length, and sampling rate (NO GRID SEARCH NEEDED)

    dataDir (str): directory with data for SVM
    fileKeyword (str): keyword to match in file names to pull relevant data
    testingRatio (number): the fraction of data to set aside for testing during training/testing split
    c (number): desired c for SVM
    degree (number): desired degree for SVM
    kernel (str): desired kernel for SVM
    shape (str): desired shape for SVM
    frameSeq (number): number of frames in a sequence (flattened together to make 1 sample for the SVM)
    samplingRate (number): rate of samples to collect (i.e. if samplingRate = 5, take every 5th row)
    jobFileName (str): desired name of output SVM joblib file
    
    Return:
    None
    Save SVM to joblib file and frameSequence-sampling combo to txt file
    '''
    
    # Get formula for sampling data based on sampling rate
    samplingFormula = lambda x: x % samplingRate != 0

    # Read data from given directory based on sampling formula and keyword to search for in file names
    fullDF = read_data(dataDir, samplingFormula, fileKeyword)
    
    # Split training/testing data by 70:30
    trainingDF, testingDF = split_data(fullDF, testingRatio, frameSeq)

    # Preprocess training and testing data
    x_train, y_train, training_distribution = preprocess_data(trainingDF, frameSeq, False)
    x_test, y_test, testing_distribution = preprocess_data(testingDF, frameSeq, False)

    # Print class distributions, if desired
    # print("Class distributions:")
    # print("Training (stable/unstable):")
    # print(training_distribution)
    # print("Testing (stable/unstable):")
    # print(testing_distribution)

    # Train SVM and save to joblib
    create_svm(x_train, y_train, x_test, y_test, svc_c=c, svc_degree=degree, svc_kernel=kernel, svc_shape=shape, seed=10, saveSVM=True, jobFileName=jobFileName)

    # Save frameSeq and samplingRate parameters for SVM
    save_params(frameSeq, samplingRate, "frameSequence-samplingRate.txt")



def main():
    dataDirectory = "./Movenet_with_labels"
    fileKeyword = "Front"
    testingRatio = 0.3
    iterations = 10  # number of splits to test SVM against and calculate average anamoly metrics
    jobFileName = "./svm.joblib"

    best_c, best_degree, best_kernel, best_shape, best_frameSeq, best_samplingRate = find_best_parameters(dataDir=dataDirectory, fileKeyword=fileKeyword, testingRatio=testingRatio, iterations=iterations, gridSearch=True)
    print("FINAL best hyperparams: ", best_c, best_degree, best_kernel, best_shape)
    print("FINAL best frame params: ", best_frameSeq, best_samplingRate)
    train_final_SVM(dataDir=dataDirectory, fileKeyword=fileKeyword, testingRatio=testingRatio, c=best_c, degree=best_degree, kernel=best_kernel, shape=best_shape, frameSeq=best_frameSeq, samplingRate=best_samplingRate, jobFileName=jobFileName)
    

if __name__ == "__main__":
    main()
