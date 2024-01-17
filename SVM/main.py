import svm

def main():
    '''
    Main function to run the SVM.
        Read data, split for training/testing, expand, and flatten spatially (row-wise) and temporally (column-wise)
        for input for spatial and temporal SVM's.
    '''
    dataDir = "./" # directory with data for SVM
    fileKeyword = "Front" # keyword in desired data file names
    samplingRate = 5 # rate to sample data
    samplingFormula = lambda x: x % samplingRate != 0
    frameSeq = 64 # frame sequence length to group data for SVM input
    testingRatio = 0.3 # testing/training data split

    # Read all data from CSV's in specified directory
    fullDF = svm.read_data(dataDir, samplingFormula, fileKeyword)

    # Split training/testing data by 70:30
    trainingDF, testingDF = svm.split_data(fullDF, testingRatio)

    # Temporal SVM
    x_train_t, y_train_t = svm.modify_input_data(trainingDF, frameSeq, "t")
    x_test_t, y_test_t = svm.modify_input_data(testingDF, frameSeq, "t")
    #svm.find_best_SVM(x_train_t, y_train_t, x_test_t, y_test_t, seed=10, nameExt="_temporal_"+str(frameSeq))
    # ^ Uncomment to find best parameters for temporally flattened data of <frameSeq>
    svm.create_svm(x_train_t, y_train_t, x_test_t, y_test_t, svc_c=10, svc_degree=1, svc_kernel='rbf', svc_shape='ovo', seed=10, nameExt="temporal_"+str(frameSeq))


if __name__ == "__main__":
    main()
