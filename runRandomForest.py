import pandas as pd

# Import Custom Libraries
import randomForestRegressor as rfg
import randomForestClassifier as rfc
import dataPrep as dp


def randomForestDiabetes():
    """
    Runs Random Forest on Pima Indians Diabetes dataset.
    """
    print("\n\nRandom Forest on Pima Indians Diabetes dataset\n")

    # Source file location
    file_orig = "data/pima-indians-diabetes.csv"

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=2, cols_to_encode=[9,10])

    # Initialize random forest object
    rfDiab = rfc.runRandomForest(file_loc,False, forest_size=100, random_seed=0, max_depth=25)

    # Train random forest model
    rfDiab.run()

def randomForestBreastCancer():
    """
    Runs Random Forest on Wisconsin Breast Prognostic dataset.
    """
    print("\n\nRandom Forest on Wisconsin Breast Prognostic dataset\n")

    # Source file location
    file_orig = "data/Wisconsin_breast_prognostic.csv"
    
    # File already formatted
    file_loc = file_orig

    # Initialize random forest object
    rfObjBreastCancer = rfc.runRandomForest(file_loc,False, forest_size=100, random_seed=0, max_depth=25)

    # Train random forest model
    rfObjBreastCancer.run()

def randomForestReg():
    """
    Runs Random Forest on Cars.com dataset.
    """
    print("\n\nRandom Forest on Cars.com dataset\n")

    # Source file location
    file_orig = "data/output_May-06-2024_cleaned"

    df = pd.read_csv(file_orig+".csv")
    df = df[['Miles', 'Stock', 'Year', 'Sub_Model','Price']]
    file_orig+="_colsSelected_.csv"
    df.to_csv(file_orig, index=False)

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=4, cols_to_encode=[1,2,3])


    # Initialize random forest object
    rfObj = rfg.runRandomForest(file_loc, forest_size=30, random_seed=0, max_depth=15)

    # Train random forest model
    randomForest,stats = rfObj.run()


randomForestDiabetes()
randomForestBreastCancer()
randomForestReg()