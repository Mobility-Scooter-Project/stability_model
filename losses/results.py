import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_metrics(positive_arr, negative_arr):
    # Combine positive and negative arrays into a single score array
    y_scores = list(positive_arr) + list(negative_arr)
    
    # Create true labels: 1 for positive and 0 for negative
    y_true = [1] * len(positive_arr) + [0] * len(negative_arr)
    
    # Compute ROC AUC
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # Compute PR AUC (average precision score)
    pr_auc = average_precision_score(y_true, y_scores)
    
    return roc_auc, pr_auc


# Function to read file and return np array
def read_file_as_np_array(filename):
    with open(filename, 'r') as f:
        # 1 - restoration loss to get stability score
        data = [(1-float(line.strip())) for line in f]
    return np.array(data)

# lstm -> vector -> lstm
unstable = read_file_as_np_array('autoencoder_02-00_1.csv')
stable= read_file_as_np_array('autoencoder_02-00_2.csv')
print(compute_metrics(stable, unstable))
# (roc auc, pr auc)
# (0.844039813338059, 0.9929606917650728)

# conv1d -> vector -> lstm
unstable = read_file_as_np_array('autoencoder_06-02_1.csv')
stable= read_file_as_np_array('autoencoder_06-02_2.csv')
print(compute_metrics(stable, unstable))
# (roc auc, pr auc)
# (0.8207028514046059, 0.9916645699293452)
