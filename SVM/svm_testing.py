from joblib import load
import pandas as pd
import numpy as np

def read_params(paramsFile):
    """
    Read the parameters for the input modifying functions
    
    Parameters:
    paramsFile (str): Path of file with threshold number

    Return:
    int: frameSeq as an int
    int: samplingRate as an int
    """

    # Read file
    with open(paramsFile, 'r') as file:
        for line in file:
            frameSeq, samplingRate = line.strip().split(',')

    return int(frameSeq), int(samplingRate)


def read_data(coordinates_arr, samplingRate):
    """
    Convert given array to a dataframe but only take every <samplingRate>th row to downscale data
    
    Parameters:
    coordinates_arr (2D array): A 2D array of coordinates to be sampled and converted to a dataframe
    samplingRate (int): The rate to sample rows from coordinates_arr

    Return:
    df (DataFrame): Newly converted coordinates dataframe
    """

    df = pd.DataFrame(coordinates_arr[::samplingRate])
    return df



# MODIFY INPUT FUNCTIONS
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


def flatten_input(df, frameSeq):
    """
    Flatten input for ML model w/ column-wise concatenation; demonstrates temporal change (how EACH COORDINATE changes over time)
    
    Parameters:
    df (DataFrame): dataframe to flatten
    frameSeq (number): length of desired frame sequence

    Return:
    df_flattened (numPy arr): flattened df
    """

    # Reshape the DataFrame by grouping rows into chunks of frameSeq
    df_groups = df.groupby(np.arange(len(df)) // frameSeq)

    # Apply a function to flatten each group column-wise
    df_flattened = df_groups.apply(lambda x: x.values.flatten('F'))

    # Reset index to create a flattened DataFrame
    df_flattened = df_flattened.reset_index(drop=True)

    # Convert pandas series to numpy arrays
    df_flattened = np.array(df_flattened.tolist())

    # Write flattened df to CSV, if desired
    # df_flattened.tofile("./flat_data.csv", sep=',')

    # Return flattened feat matrix
    return df_flattened


def modify_input_data(df, frameSeq):
    """
    Round a dataframe and flatten it.
        Wrapper function for round_df() and flatten_input()
    
    Parameters:
    df (DataFrame): dataframe to preprocess
    frameSeq (number): length of frame sequence

    Return: 
    Return vars are from flatten_input():
        df_flattened (numPy arr): reduced and flattened df
    """

    # Make df a multiple of frameSeq
    df = round_df(df, frameSeq)

    # Flatten rows
    return flatten_input(df, frameSeq)
   


def convert_to_hashmap(coor_arr, predictions_arr):
    """
    Merge the coordinates and corresponding predictions numPy arrays into a hashmap

    Parameters:
    coor_arr (numPy array): Coordinates
    predictions_arr (numPy array): Predictions for coordinates

    Return:
    Hashmap: Coordinates are keys and predictions are values
    """
    
    # Decode classes; 1 = stable, 0 = unstable
    prediction_strings = np.where(predictions_arr == 1, 'stable', 'unstable')

    # Combine coordinate and prediction arrays into one hashmap
    hashmap = {tuple(key): value for key, value in zip(coor_arr, prediction_strings)}
    
    return hashmap


def classify_pose_data(coordinates_arr, svmFile, parametersFile):
    """
    Make predictions ("stable"/"unstable") on given keypoint coordinates

    Parameters:
    coordinates_arr (list of lists): A 2D array of coordinates
    svmFile (str): The path to the SVM job file to load
    parametersFile (str): The path to the text file with the frameSeq and samplingRate parameters

    Return: 
    Hashmap: Coordinates are groups of length <frameSeq> and each group is mapped to a prediction ("stable" or "unstable")
    """
    # Load variables, data, and SVM
    frameSeq, samplingRate = read_params(parametersFile)
    testingDF = read_data(coordinates_arr, samplingRate)
    svm_model = load(svmFile)
    
    # Transform input data
    modified_data = modify_input_data(testingDF, frameSeq)

    # Make predictions with the SVM
    predictions = svm_model.predict(modified_data)

    # Convert to hashmap
    final_hashmap = convert_to_hashmap(modified_data, predictions)
    
    return final_hashmap


def main():
    sample_arr = [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
    ]
    sample_arr2 = np.genfromtxt('sample.csv', delimiter=',')

    classify_pose_data(sample_arr2, "./svm.joblib", "./svm_parameters.txt")


if __name__ == "__main__":
    main()