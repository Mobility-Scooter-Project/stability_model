import pandas as pd
import json

def read_threshold(thresholdFile):
    """
    Read the upper threshold for the statistics model from a file
    
    Parameters:
    thresholdFile (str): Path of file with threshold number

    Return:
    float: threshold as a float
    """

    with open(thresholdFile, 'r') as file:
        # Read file
        content = file.read().strip()
        
        # Convert content to a number
        try:
            threshold = float(content)
            # print("Threshold:", threshold)
            return threshold
        except ValueError:
            print("Error: Cannot read a number from this file")


def classify_value(value, upperLimit):
    """
    Classify value as "stable" or "unstable" based on limits.
    To be used in conjuction with map() function.

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
    

def convert_to_hashmap(losses_df, classes_df):
    """
    Merge the losses and corresponding classifications dataframes into a hashmap
    (Optional) convert the hashmap to a string

    Parameters:
    losses_df (DataFrame): Losses
    classes_df (DataFrame): Classifictions for losses

    Return:
    Hashmap: Losses are keys and classes are values
    """

    merged_df = pd.concat([losses_df, classes_df], axis=1)

    hashmap = merged_df.set_index("Losses").to_dict()["Classes"]

    # # Convert hashmap to string
    # hashmap_str = json.dumps(hashmap)
    # print(hashmap_str)

    return hashmap


def classify_losses(losses_arr, thresholdFile):
    """
    Classify an array of losses based on an upper threshold.

    Parameters:
    losses_arr (arr[]): Losses array
    thresholdFile (str): Path to file with threshold

    Return:
    Hashmap: Losses-classifications hashmap
    """

    # Read threshold from file
    upper_threshold = read_threshold(thresholdFile)

    # Convert losses array to dataframe for classify_value function
    losses_df = pd.DataFrame({"Losses": losses_arr})

    # Classify losses and get a classifications dataframe
    classifications = losses_df.map(lambda x: classify_value(x, upper_threshold))
    classes_df = pd.DataFrame(classifications)
    classes_df.rename(columns={'Losses': 'Classes'}, inplace=True)

    # Convert losses and classes to a hashmap
    return convert_to_hashmap(losses_df, classes_df)



# Can run model in main or import functions to another file
def main():
    # Input sample loss values
    sample_arr = []
    thresholdFile = "threshold_m1.txt"

    return classify_losses(sample_arr, thresholdFile)


if __name__ == "__main__":
    main()
