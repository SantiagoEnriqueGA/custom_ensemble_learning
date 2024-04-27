import randomForest as rf
import dataPrep as dp

def randomForestDiabetes():
    """
    Runs Random Forest on Pima Indians Diabetes dataset.
    """
    print("\n\nRandom Forest on Pima Indians Diabetes dataset\n")

    # Source file location
    file_orig = "pima-indians-diabetes.csv"

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=2, cols_to_encode=[9,10])

    # Initialize random forest object
    rfDiab = rf.runRandomForest(file_loc,False, forest_size=100, random_seed=0, max_depth=25)

    # Train random forest model
    rfDiab.run()

def randomForestBreastCancer():
    """
    Runs Random Forest on Wisconsin Breast Prognostic dataset.
    """
    print("\n\nRandom Forest on Wisconsin Breast Prognostic dataset\n")

    # Source file location
    file_orig = "Wisconsin_breast_prognostic.csv"
    
    # File already formatted
    file_loc = file_orig

    # Initialize random forest object
    rfObjBreastCancer = rf.runRandomForest(file_loc,False, forest_size=100, random_seed=0, max_depth=25)

    # Train random forest model
    rfObjBreastCancer.run()

randomForestBreastCancer()
randomForestDiabetes()