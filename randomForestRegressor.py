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

    num_trees = 0
    decision_trees = []
    bootstraps_datasets = []
    bootstraps_labels = []
    max_depth = 10

    def __init__(self, num_trees, max_depth):
        """
        Initializes the RandomForest object.

        Args:
            num_trees (int): The number of decision trees in the random forest.
            max_depth (int): The maximum depth of each decision tree.
        """
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.decision_trees = [DecisionTreeRegressor(max_depth) for i in range(num_trees)]
        self.bootstraps_datasets = []
        self.bootstraps_labels = []

    def _bootstrapping(self, XX, n):
        """
        Performs bootstrapping on the dataset.

        Args:
            XX (list): The dataset.
            n (int): The number of samples to be selected.

        Returns:
            tuple: A tuple containing the bootstrapped dataset and the corresponding labels.
        """
        sample_indices = np.random.choice(len(XX), size=n, replace=True)
        sample = [XX[i][:-1] for i in sample_indices]
        labels = [XX[i][-1] for i in sample_indices]
        return (sample, labels)

    def bootstrapping(self, XX):
        """
        Initializes the bootstrapped datasets for each tree.

        Args:
            XX (list): The dataset.
        """
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        """
        Fits the decision trees to the bootstrapped datasets.
        """
        for i in range(self.num_trees):
            tree = self.decision_trees[i]
            dataset = self.bootstraps_datasets[i]
            labels = self.bootstraps_labels[i]
            self.decision_trees[i] = tree.learn(dataset, labels)

    def voting(self, X):
        """
        Performs voting to predict the target values for the input records.

        Args:
            X (list): The input records.

        Returns:
            list: The predicted target values for the input records.
        """
        y = []

        for record in X:
            predictions = []

            for i, dataset in enumerate(self.bootstraps_datasets):
                # For unbiased estimate, only consider the trees that did not train on the record
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    prediction = DecisionTreeRegressor.predict(OOB_tree, record)
                    predictions.append(prediction)

            if len(predictions) > 0:
                mean_prediction = np.mean(predictions)
                y.append(mean_prediction)
            else:
                # If the record is not out-of-bag (OOB), use all trees for prediction
                for i in range(self.num_trees):
                    tree = self.decision_trees[i]
                    predictions.append(DecisionTreeRegressor.predict(tree, record))
                y.append(np.mean(predictions))


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

    random_seed = 0
    forest_size = 10
    max_depth = 10
    display = False
    X = list()
    y = list()
    XX = list()  # Contains both data features and data labels
    numerical_cols = 0

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
        self.reset()

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.forest_size = forest_size
        self.max_depth = max_depth
        
        # Get the indices of numeric attributes (columns)
        self.numerical_cols = set()
        with open(file_loc, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for i in range(len(headers)):
                try:
                    # Try to convert the first row (excluding label) to float
                    float(next(reader)[i])
                    self.numerical_cols.add(i)
                except ValueError:
                    continue

        # Loading data set
        print("reading the data")
        try:
            with open(file_loc) as f:
                next(f, None)
                for line in csv.reader(f, delimiter=","):
                    xline = []
                    for i in range(len(line)):
                        if i in self.numerical_cols:
                            xline.append(ast.literal_eval(line[i]))
                        else:
                            xline.append(line[i])

                    self.X.append(xline[:-1])
                    self.y.append(xline[-1])
                    self.XX.append(xline[:])
        except FileNotFoundError:
            print(f"File {file_loc} not found.")
            return None, None
        
    def reset(self):
        """
        Resets the random forest object.
        """
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
        # start time
        start = datetime.now()

        # Initializing a random forest.
        randomForest = RandomForest(self.forest_size,self.max_depth)
 
        # Creating the bootstrapping datasets
        print("creating the bootstrap datasets")
        randomForest.bootstrapping(self.XX)

        # Building trees in the forest
        print("fitting the forest")
        randomForest.fitting()

        # Calculating an unbiased error estimation of the random forest based on out-of-bag (OOB) error estimate.
        y_predicted = randomForest.voting(self.X)

        # Mean Squared Error (MSE)
        mse = np.mean((np.array(y_predicted) - np.array(self.y)) ** 2)
        print("MSE: %.4f" % mse)

        # R^2 Score
        ssr = np.sum((np.array(y_predicted) - np.array(self.y)) ** 2)
        sst = np.sum((np.array(self.y) - np.mean(self.y)) ** 2)
        r2 = 1 - (ssr / sst)
        print("R^2 Score: %.4f" % r2)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((np.array(self.y) - np.array(y_predicted)) / np.array(self.y))) * 100
        print("MAPE: %.4f%%" % mape)

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(np.array(self.y) - np.array(y_predicted)))
        print("MAE: %.4f" % mae)

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((np.array(y_predicted) - np.array(self.y)) ** 2))
        print("RMSE: %.4f" % rmse)

        stats = {
            "MSE": mse,
            "R^2": r2,
            "MAPE": mape,
            "MAE": mae,
            "RMSE": rmse
        }

        # End time
        print("Execution time: " + str(datetime.now() - start))

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

