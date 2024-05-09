import pandas as pd

# Import Custom Classes
import gradientBoostedRegressor as gbr
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
    rfDiab = rfc.runRandomForest(file_loc,False, forest_size=10, random_seed=0, max_depth=25)

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
    rfObjBreastCancer = rfc.runRandomForest(file_loc,False, forest_size=10, random_seed=0, max_depth=25)

    # Train random forest model
    rfObjBreastCancer.run()

def randomForestCarsReg():
    """
    Runs Random Forest on Cars.com dataset.
    """
    print("\n\nRandom Forest on Cars.com dataset\n")

    df = pd.read_csv("data/output_May-06-2024_cleaned.csv")
    df = df[['Miles', 'Stock', 'Year', 'Sub_Model','Price']]
    df.to_csv("data/carsDotCom.csv", index=False)

    # Source file location
    file_orig = "data/carsDotCom.csv"

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=4, cols_to_encode=[1,2,3])


    # Initialize random forest object
    rfObj = rfg.runRandomForest(file_loc, forest_size=3, random_seed=0, max_depth=10)

    # Train random forest model
    randomForest,stats = rfObj.run()

def gbtrCarsReg():
    """
    Runs Gradient Boosted Decision Trees on the Cars.com dataset.
    """
    # Source file location
    file_orig = "data/carsDotCom.csv"

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=4, cols_to_encode=[1,2,3])

    # Initialize GBDT object
    gbdtDiab = gbr.gradientBoostedRegressor(file_loc, num_trees=5, random_seed=0, max_depth=10)

    # Train GBDT model
    gbdtDiab.fit(stats=True)

    # Predict target values
    predictions = gbdtDiab.predict()

    # Get stats
    stats = gbdtDiab.get_stats(predictions)
    print(stats)





randomForestDiabetes()
randomForestBreastCancer()
randomForestCarsReg()
gbtrCarsReg()


