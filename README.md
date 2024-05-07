# Custom Random Forest Classifier and Regressor Project

## Overview

This project implements a custom random forest classifier for binary classification tasks and a custom random forest regressor for regression tasks. It consists of the following main Python files:

- `dataPrep.py`: Contains a class `DataPrep` for preparing data for machine learning models, including custom functions for one-hot encoding non-numerical columns and writing data to CSV files.
  
- `randomForestClassifier.py`: Implements a custom random forest classifier along with utility functions for computing entropy, partitioning classes, and calculating information gain.
  
- `randomForestRegressor.py`: Implements a modified version of `randomForestClassifier.py` updated to perform regression tasks.

- `runRandomForest.py`: Updated to run `randomForestClassifier.py` on the Pima Indians Diabetes dataset and Wisconsin Breast Prognostic dataset, and `randomForestRegressor.py` on `output_May-06-2024_cleaned.csv`.

## Included CSV Files

- `Wisconsin_breast_prognostic.csv`: Dataset used for the Wisconsin Breast Prognostic dataset analysis.
  
- `pima-indians-diabetes.csv`: Dataset used for the Pima Indians Diabetes dataset analysis.

- `output_May-06-2024_cleaned.csv`: Car listing data scraped from cars.com via my repository "used_car_price_visualization".

## `dataPrep.py`

### Class: `DataPrep`

- `one_hot_encode(df, cols)`: One-hot encodes non-numerical columns in a DataFrame.
  
- `write_data(df, csv_file)`: Writes the DataFrame to a CSV file.
  
- `prepare_data(csv_file, label_col_index, cols_to_encode=[], write_to_csv=True)`: Prepares the data by loading a CSV file, one-hot encoding non-numerical columns, and optionally writing the prepared data to a new CSV file.

## `randomForestClassifier.py`

### Classes: `Utility`, `DecisionTree`, `DecisionTreeWithInfoGain`, `RandomForest`, `RandomForestWithInfoGain`

- `Utility`: Utility class for computing entropy, partitioning classes, and calculating information gain.
  
- `DecisionTree`: Represents a decision tree for classification tasks.
  
- `DecisionTreeWithInfoGain`: Extends `DecisionTree` to use information gain for splitting.
  
- `RandomForest`: Implements a custom random forest classifier with bootstrapping and voting mechanisms.
  
- `RandomForestWithInfoGain`: Extends `RandomForest` to use information gain for splitting.

## `randomForestRegressor.py`

### Classes: `Utility`, `DecisionTree`, `RandomForest`,

- `Utility`: Utility class for computing variance, partitioning classes, and calculating information gain.
  
- `DecisionTreeRegressor`: Represents a decision tree for regression tasks.
    
- `RandomForest`: Implements a custom random forest classifier with bootstrapping and voting mechanisms.
  

## `runRandomForest.py`

This file contains functions to run the custom random forest classifier on two datasets:

- `randomForestDiabetes()`: Runs random forest classifier on the Pima Indians Diabetes dataset.
  
- `randomForestBreastCancer()`: Runs random forest classifier on the Wisconsin Breast Prognostic dataset.

- `randomForestReg()`: Runs the random forest regressor on the cars.com, `output_May-06-2024_cleaned.csv` dataset.

