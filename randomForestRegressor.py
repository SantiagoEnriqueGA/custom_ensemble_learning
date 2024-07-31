"""
This module contains the implementation of a Random Forest Regressor.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- runRandomForest: A class that runs the Random Forest algorithm.
"""

# Importing the required libraries
import csv
import numpy as np
import ast
from datetime import datetime
from math import log, floor, ceil
import random
import matplotlib.pyplot as plt
from decisionTreeRegressor import Utility, DecisionTreeRegressor

class RandomForest(object):
    """
    A class representing a Random Forest model.

    Attributes:
        num_trees (int): The number of decision trees in the random forest.
        decision_trees (list): A list of decision trees in the random forest.
        bootstraps_datasets (list): A list of bootstrapped datasets for each tree.
        bootstraps_labels (list): A list of corresponding labels for each bootstrapped dataset.
        max_depth (int): The maximum depth of each decision tree.

    Methods:
        __init__(num_trees, max_depth): Initializes the RandomForest object.
        _bootstrapping(XX, n): Performs bootstrapping on the dataset.
        bootstrapping(XX): Initializes the bootstrapped datasets for each tree.
        fitting(): Fits the decision trees to the bootstrapped datasets.
        voting(X): Performs voting to predict the target values for the input records.
        user(): Returns the user's GTUsername.
    """
    # Initialize class variables
    num_trees = 0               # Number of decision trees in the random forest
    decision_trees = []         # List of decision trees in the random forest
    bootstraps_datasets = []    # List of bootstrapped datasets for each tree
    bootstraps_labels = []      # List of true class labels corresponding to records in the bootstrapped datasets
    max_depth = 10              # Maximum depth of each decision tree

    def __init__(self, num_trees, max_depth):
        """
        Initializes the RandomForest object.

        Args:
            num_trees (int): The number of decision trees in the random forest.
            max_depth (int): The maximum depth of each decision tree.
        """
        self.num_trees = num_trees  # Set the number of trees
        self.max_depth = max_depth  # Set the maximum depth

        self.decision_trees = [DecisionTreeRegressor(max_depth) for i in range(num_trees)]  # Initialize the decision trees
        
        self.bootstraps_datasets = []   # Initialize the list of bootstrapped datasets
        self.bootstraps_labels = []     # Initialize the list of corresponding labels

    def _bootstrapping(self, XX, n):
        """
        Performs bootstrapping on the dataset.

        Args:
            XX (list): The dataset.
            n (int): The number of samples to be selected.

        Returns:
            tuple: A tuple containing the bootstrapped dataset and the corresponding labels.
        """
        sample_indices = np.random.choice(len(XX), size=n, replace=True)    # Randomly select indices with replacement
        
        sample = [XX[i][:-1] for i in sample_indices]   # Get the features of the selected samples
        labels = [XX[i][-1] for i in sample_indices]    # Get the labels of the selected samples
        
        return (sample, labels)

    def bootstrapping(self, XX):
        """
        Initializes the bootstrapped datasets for each tree.

        Args:
            XX (list): The dataset.
        """
        for i in range(self.num_trees):                                 # For each tree
            data_sample, data_label = self._bootstrapping(XX, len(XX))  # Perform bootstrapping on the dataset
            self.bootstraps_datasets.append(data_sample)                # Add the bootstrapped dataset to the list
            self.bootstraps_labels.append(data_label)                   # Add the corresponding labels to the list

    def fitting(self):
        """
        Fits the decision trees to the bootstrapped datasets.
        """
        for i in range(self.num_trees):             # For each tree
            tree = self.decision_trees[i]           # Get the current tree
            dataset = self.bootstraps_datasets[i]   # Get the bootstrapped dataset
            labels = self.bootstraps_labels[i]      # Get the corresponding labels
            
            self.decision_trees[i] = tree.learn(dataset, labels)    # Fit the tree to the bootstrapped dataset

    def voting(self, X):
        """
        Performs voting to predict the target values for the input records.

        Args:
            X (list): The input records.

        Returns:
            list: The predicted target values for the input records.
        """
        y = []
        for record in X:        # For each record
            predictions = []
            for i, dataset in enumerate(self.bootstraps_datasets):  # For each bootstrapped dataset
                
                # Records not in the dataset are considered out-of-bag (OOB) records, which can be used for voting
                if record not in dataset:               # If the record is not in the dataset
                    OOB_tree = self.decision_trees[i]   # Get the decision tree corresponding to the dataset
                    prediction = DecisionTreeRegressor.predict(OOB_tree, record)    # Predict the target value for the record
                    predictions.append(prediction)      # Add the prediction to the votes list

            # Calculate the mean prediction for the record
            if len(predictions) > 0:                    # If there are predictions
                mean_prediction = np.mean(predictions)  # Calculate the mean prediction
                y.append(mean_prediction)               # Add the mean prediction to the list
            
            else:   # If the record is not out-of-bag (OOB), use all trees for prediction
                for i in range(self.num_trees):     # For each tree
                    tree = self.decision_trees[i]   # Get the current tree
                    predictions.append(DecisionTreeRegressor.predict(tree, record)) # Predict the target value for the record
                y.append(np.mean(predictions))      # Add the mean prediction to the list

        return y

class runRandomForest(object):
    """
    A class that represents a random forest algorithm.

    Attributes:
        random_seed (int): The random seed for reproducibility.
        forest_size (int): The number of trees in the random forest.
        max_depth (int): The maximum depth of each decision tree in the random forest.
        display (bool): Whether to display additional information about info gain.
        X (list): The list of data features.
        y (list): The list of data labels.
        XX (list): The list that contains both data features and data labels.
        numerical_cols (int): The number of numeric attributes (columns).

    Methods:
        __init__(self, file_loc, display=False, forest_size=5, random_seed=0, max_depth=10):
            Initializes the random forest object.

        reset(self):
            Resets the random forest object.

        run(self):
            Runs the random forest algorithm.

    Example:
        randomForest, accuracy = runRandomForest('data.csv', display=True, forest_size=10, random_seed=42)
    """
    # Initialize class variables
    random_seed = 0     # Random seed for reproducibility
    forest_size = 10    # Number of trees in the random forest
    max_depth = 10      # Maximum depth of each decision tree
    display = False     # Flag to display additional information about info gain
    
    X = list()          # Data features
    y = list()          # Data labels
    XX = list()         # Contains both data features and data labels
    numerical_cols = 0  # Number of numeric attributes (columns)

    def __init__(self, file_loc, forest_size=5, random_seed=0, max_depth=10):
        """
        Initializes the random forest object.

        Args:
            file_loc (str): The file location of the dataset.
            display (bool, optional): Whether to display additional information about info gain. Defaults to False.
            forest_size (int, optional): The number of trees in the random forest. Defaults to 5.
            random_seed (int, optional): The random seed for reproducibility. Defaults to 0.
            max_depth (int, optional): The maximum depth of each decision tree in the random forest. Defaults to 10.
        """
        self.reset()    # Reset the random forest object

        self.random_seed = random_seed  # Set the random seed for reproducibility
        np.random.seed(random_seed)     # Set the random seed for NumPy

        self.forest_size = forest_size  # Set the number of trees in the random forest
        self.max_depth = max_depth      # Set the maximum depth of each decision tree
        
        self.numerical_cols = set()         # Initialize the set of indices of numeric attributes (columns)
        with open(file_loc, 'r') as f:      # Open the file in read mode
            reader = csv.reader(f)          # Create a CSV reader
            headers = next(reader)          # Get the headers of the CSV file
            for i in range(len(headers)):   # Loop over the indices of the headers
                try:
                    float(next(reader)[i])      # If successful, add the index to the set of numerical columns
                    self.numerical_cols.add(i)  # Add the index to the set of numerical columns
                except ValueError:
                    continue

        print("reading the data")
        try:
            with open(file_loc) as f:                       # Open the file
                next(f, None)                               # Skip the header
                for line in csv.reader(f, delimiter=","):   # Read the file line by line
                    xline = []                  
                    for i in range(len(line)):              # Loop over the indices of the line
                        if i in self.numerical_cols:                # If the index is in the set of numerical columns
                            xline.append(ast.literal_eval(line[i])) # Append the value to the input data features
                        
                        else:                                       # If the index is not in the set of numerical columns
                            xline.append(line[i])                   # Append the value to the input data features    

                    self.X.append(xline[:-1])   # Append the input data features to the list of input data features
                    self.y.append(xline[-1])    # Append the target value to the list of target values
                    self.XX.append(xline[:])    # Append the input data features and target value to the list of input data features and target values
        except FileNotFoundError:
            print(f"File {file_loc} not found.")
            return None, None
        
    def reset(self):
        """
        Resets the random forest object.
        """
        # Reset the random forest object
        self.random_seed = 0
        self.forest_size = 10
        self.max_depth = 10
        self.display = False
        self.X = list()
        self.y = list()
        self.XX = list()
        self.numerical_cols = 0

    def run(self):
        """
        Runs the random forest algorithm.

        Returns:
            tuple: A tuple containing the random forest object and the accuracy of the random forest algorithm.

        Raises:
            FileNotFoundError: If the file specified by file_loc does not exist.

        Notes:
            - The file should have the following format:
                - Each row represents a data point (record).
                - The last column represents the class label.
                - The remaining columns represent the features (attributes).
                - Features are numerical and class labels are binary (0 or 1).
            - The random seed is used to initialize the random number generator for reproducibility.
            - The random forest object contains the trained random forest model.
            - The accuracy is calculated as the ratio of correctly predicted labels to the total number of labels.
        """
        start = datetime.now()  # Start time

        randomForest = RandomForest(self.forest_size,self.max_depth)    # Initialize the random forest object
 
        print("creating the bootstrap datasets")
        randomForest.bootstrapping(self.XX)         # Create the bootstrapped datasets

        print("fitting the forest")
        randomForest.fitting()                      # Fit the decision trees to the bootstrapped datasets
        y_predicted = randomForest.voting(self.X)   # Predict the target values for the input records
        
        print("Execution time: " + str(datetime.now() - start))

        # Calculate evaluation metrics
        mse = np.mean((np.array(y_predicted) - np.array(self.y)) ** 2)  # Calculate the mean squared error (MSE): mean((y_true - y_pred)^2)
        ssr = np.sum((np.array(y_predicted) - np.array(self.y)) ** 2)   # Calculate the sum of squared residuals (SSR): sum((y_true - y_pred)^2)
        sst = np.sum((np.array(self.y) - np.mean(self.y)) ** 2)         # Calculate the total sum of squares (SST): sum((y_true - mean(y_true))^2)
        r2 = 1 - (ssr / sst)                                            # Calculate the R^2 score: 1 - (SSR / SST)
        mape = np.mean(np.abs((np.array(self.y) - 
                               np.array(y_predicted)) / np.array(self.y))) * 100    # Calculate the mean absolute percentage error (MAPE): mean(abs((y_true - y_pred) / y_true)) * 100
        mae = np.mean(np.abs(np.array(self.y) - np.array(y_predicted)))             # Mean Absolute Error (MAE): mean(abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((np.array(y_predicted) - np.array(self.y)) ** 2))    # Root Mean Squared Error (RMSE): sqrt(mean((y_true - y_pred)^2))

        # Print the evaluation metrics
        print("MSE: %.4f" % mse)
        print("R^2 Score: %.4f" % r2)
        print("MAPE: %.4f%%" % mape)
        print("MAE: %.4f" % mae)
        print("RMSE: %.4f" % rmse)

        stats = {"MSE": mse,"R^2": r2,"MAPE": mape,"MAE": mae,"RMSE": rmse}

        return randomForest,stats


# Example Usage:
# # Source file location
# file_loc = "data/carsDotCom_prepared.csv"

# # Initialize random forest object
# rfObj = runRandomForest(file_loc, forest_size=10, random_seed=0, max_depth=10)
# # Train random forest model
# randomForest,stats = rfObj.run()

# forests = [3,5,10,25,50]
# depths = [5,10,15,20,25]
# run_stats = {
#     "Forest Size" : None,
#     "Max Depth" : None,
#     "MSE" : None,
#     "R^2" : None,
#     "MAPE" : None,
#     "MAE" : None,
#     "RMSE" : None
# }
# for forest in forests:
#     for depth in depths:
#         print(f"Random Forest with {forest} trees and max depth of {depth}")

#         rfObj = runRandomForest(file_loc, forest_size=forest, random_seed=0, max_depth=depth)
#         randomForest,stats = rfObj.run()
        
#         print(stats)
#         print("\n")
        
#         run_stats["Forest Size"] = forest
#         run_stats["Max Depth"] = depth
#         run_stats['MSE'] = stats['MSE']
#         run_stats['R^2'] = stats['R^2']
#         run_stats['MAPE'] = stats['MAPE']
#         run_stats['MAE'] = stats['MAE']
#         run_stats['RMSE'] = stats['RMSE']

