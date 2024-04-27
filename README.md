# Custom Random Forest Classifier Project

## Overview

This project implements a custom random forest classifier for binary classification tasks. It consists of three main Python files:

- `dataPrep.py`: Contains a class `DataPrep` for preparing data for machine learning models, including custom functions for one-hot encoding non-numerical columns and writing data to CSV files.

- `randomForest.py`: Implements a custom random forest classifier along with utility functions for computing entropy, partitioning classes, and calculating information gain.

- `runRandomForest.py`: Runs the custom random forest classifier on two datasets: Pima Indians Diabetes dataset and Wisconsin Breast Prognostic dataset.

## Included CSV Files

- `Wisconsin_breast_prognostic.csv`: Dataset used for the Wisconsin Breast Prognostic dataset analysis.

- `pima-indians-diabetes.csv`: Dataset used for the Pima Indians Diabetes dataset analysis.

## `dataPrep.py`

### Class: `DataPrep`

- `one_hot_encode(df, cols)`: One-hot encodes non-numerical columns in a DataFrame.
- `write_data(df, csv_file)`: Writes the DataFrame to a CSV file.
- `prepare_data(csv_file, label_col_index, cols_to_encode=[], write_to_csv=True)`: Prepares the data by loading a CSV file, one-hot encoding non-numerical columns, and optionally writing the prepared data to a new CSV file.

## `randomForest.py`

### Classes: `Utility`, `DecisionTree`, `DecisionTreeWithInfoGain`, `RandomForest`, `RandomForestWithInfoGain`

- `Utility`: Utility class for computing entropy, partitioning classes, and calculating information gain.
- `DecisionTree`: Represents a decision tree for classification tasks.
- `DecisionTreeWithInfoGain`: Extends `DecisionTree` to use information gain for splitting.
- `RandomForest`: Implements a custom random forest classifier with bootstrapping and voting mechanisms.
- `RandomForestWithInfoGain`: Extends `RandomForest` to use information gain for splitting.

## `runRandomForest.py`

This file contains functions to run the custom random forest classifier on two datasets:

- `randomForestDiabetes()`: Runs random forest on the Pima Indians Diabetes dataset.
- `randomForestBreastCancer()`: Runs random forest on the Wisconsin Breast Prognostic dataset.
