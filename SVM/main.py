import svm

def main():
    '''
    Main function to run the SVM.
        Read data, split for training/testing, expand, and flatten spatially (row-wise) and temporally (column-wise)
        for input for spatial and temporal SVM's.
    '''
    samplingRate = lambda x: x % 5 != 0
    frameSeq = 64 # frames

    # Read all data from CSV's in specified directory
    fullDF = svm.read_data("./", samplingRate, "Front") # can change "./" to directory with CSV's + change keyword ("Front")

    # Split training/testing data by 70:30
    trainingDF, testingDF = svm.split_data(fullDF, 0.3) # can change 0.3 to desired testing data ratio
    trainingDF.to_csv("training.csv", index=False)

    # Spatial SVM
    x_train_s, y_train_s = svm.modify_input_data(trainingDF, frameSeq, "s")
    x_test_s, y_test_s = svm.modify_input_data(testingDF, frameSeq, "s")
    #svm.find_best_SVM(x_train_s, y_train_s, x_test_s, y_test_s, seed=10, nameExt="_spatial_"+str(frameSeq))
    # ^ Uncomment to find best parameters for spatially flattened data of <frameSeq>
    svm.create_svm(x_train_s, y_train_s, x_test_s, y_test_s, svc_c=10, svc_degree=1, svc_kernel='rbf', svc_shape='ovo', seed=10, nameExt="spatial_"+str(frameSeq))

    # Temporal SVM
    x_train_t, y_train_t = svm.modify_input_data(trainingDF, frameSeq, "t")
    x_test_t, y_test_t = svm.modify_input_data(testingDF, frameSeq, "t")
    #svm.find_best_SVM(x_train_t, y_train_t, x_test_t, y_test_t, seed=10, nameExt="_temporal_"+str(frameSeq))
    # ^ Uncomment to find best parameters for temporally flattened data of <frameSeq>
    svm.create_svm(x_train_t, y_train_t, x_test_t, y_test_t, svc_c=10, svc_degree=1, svc_kernel='rbf', svc_shape='ovo', seed=10, nameExt="temporal_"+str(frameSeq))

    # TEST w/ DIF FRAME SEQ, SAMPLING RATE, SPLIT RATIO, ETC.



if __name__ == "__main__":
    main()
