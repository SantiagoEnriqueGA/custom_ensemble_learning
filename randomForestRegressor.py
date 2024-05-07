"""
This module contains the implementation of a Random Forest Regressor.

The module includes the following classes:
- Utility: A utility class for computing entropy, partitioning classes, and calculating information gain.
- DecisionTreeRegressor: A class representing a decision tree for regression.
- RandomForest: A class representing a Random Forest model.

"""



# Importing the required libraries
import csv
import numpy as np
import ast
from datetime import datetime
from math import log, floor, ceil
import random
import matplotlib.pyplot as plt

class Utility(object):
    """
    Utility class for computing entropy, partitioning classes, and calculating information gain.
    """

    def calculate_variance(self, y):
        """
        Calculate the variance of a dataset.
        Variance is used as the measure of impurity in the case of regression.
        """
        if len(y) == 0:   # empty list
            return 0
        mean = sum(y) / len(y)
        variance = sum((yi - mean) ** 2 for yi in y) / len(y)
        return variance


    def partition_classes(self, X, y, split_attribute, split_val):
        """
        Partitions the dataset into two subsets based on a given split attribute and value.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target labels.
        - split_attribute (int): The index of the attribute to split on.
        - split_val (float): The value to split the attribute on.

        Returns:
        - X_left (array-like): The subset of input features where the split attribute is less than or equal to the split value.
        - X_right (array-like): The subset of input features where the split attribute is greater than the split value.
        - y_left (array-like): The subset of target labels corresponding to X_left.
        - y_right (array-like): The subset of target labels corresponding to X_right.
        """
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y)

        # Check if X is 1D (only one feature)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Convert to a 2D array with one column

        # Use NumPy boolean indexing for partitioning
        X_left = X[X[:, split_attribute] <= split_val]
        X_right = X[X[:, split_attribute] > split_val]
        y_left = y[X[:, split_attribute] <= split_val]
        y_right = y[X[:, split_attribute] > split_val]

        return X_left, X_right, y_left, y_right


    def information_gain(self, previous_y, current_y):
        """
        Calculate the information gain from a split by subtracting the variance of
        child nodes from the variance of the parent node.
        """
        # Calculate the variance of the parent node
        parent_variance = self.calculate_variance(previous_y)

        # Calculate the variance of each child node and their weighted average
        child_variance = sum(self.calculate_variance(y) * len(y) for y in current_y) / len(previous_y)

        # The information gain is the parent's variance minus the weighted average of the child nodes' variance
        info_gain = parent_variance - child_variance

        return info_gain
    

    def best_split(self, X, y):
        """
        Finds the best attribute and value to split the data based on information gain.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target variable.

        Returns:
        - dict: A dictionary containing the best split attribute, split value, left and right subsets of X and y,
                and the information gain achieved by the split.
        """
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Randomly select a subset of attributes for splitting
        num_features = int(np.sqrt(X.shape[1]))  # Square root of total attributes
        selected_attributes = np.random.choice(X.shape[1], size=num_features, replace=False)

        # Initialize the best information gain to negative infinity, others to None
        best_info_gain = float('-inf')
        best_split_attribute = None
        best_split_val = None
        best_X_left, best_X_right, best_y_left, best_y_right = None, None, None, None

        # Iterate over each attribute in the selected subset to find the best split
        for split_attribute in selected_attributes:
            values = np.unique(X[:, split_attribute])
            for split_val in values:
                # Perform partitioning
                X_left, X_right, y_left, y_right = self.partition_classes(X, y, split_attribute, split_val)
                # Calculate information gain
                info_gain = self.information_gain(y, [y_left, y_right])
                # Update best split if info_gain is greater
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split_attribute = split_attribute
                    best_split_val = split_val
                    best_X_left, best_X_right, best_y_left, best_y_right = X_left, X_right, y_left, y_right

        # Return the best split
        return {'split_attribute': best_split_attribute,
                'split_val': best_split_val,
                'X_left': best_X_left,
                'X_right': best_X_right,
                'y_left': best_y_left,
                'y_right': best_y_right,
                'info_gain': best_info_gain}

class DecisionTreeRegressor(object):
    """
    A class representing a decision tree for regression.

    Parameters:
    - max_depth (int): The maximum depth of the decision tree.

    Methods:
    - learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
    - classify(record): Predicts the target value for a record using the decision tree.

    """

    def __init__(self, max_depth):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = max_depth


    def learn(self, X, y, par_node={}, depth=0):
        """
        Builds the decision tree based on the given training data.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target labels.
        - par_node (dict): The parent node of the current subtree (default: {}).
        - depth (int): The current depth of the subtree (default: 0).

        Returns:
        - dict: The learned decision tree.

        """
        # Convert y to a Python list
        y = y.tolist() if isinstance(y, np.ndarray) else y
        
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y, dtype=int)
        
        # Check if the node is pure (all values are the same)
        if len(set(y)) == 1:
            return {'value': y[0]}

        # Check if maximum depth is reached
        if depth >= self.max_depth:
            return {'value': np.mean(y)}

        # Get the best split using utility functions
        best_split = Utility().best_split(X, y)
        split_attribute = best_split['split_attribute']
        split_val = best_split['split_val']
        X_left = best_split['X_left']
        X_right = best_split['X_right']
        y_left = best_split['y_left']
        y_right = best_split['y_right']

        # Check if there is no further reduction in variance
        if best_split['info_gain'] == 0:
            return {'value': np.mean(y)}

        # Recursively build the left and right subtrees
        par_node = {'split_attribute': split_attribute, 'split_val': split_val}
        par_node['left'] = self.learn(X_left, y_left, depth=depth + 1)
        par_node['right'] = self.learn(X_right, y_right, depth=depth + 1)

        return par_node

    def Predict(self, record):
        """
        Predicts a given record using the decision tree.

        Parameters:
        - record: A dictionary representing the record to be classified.

        Returns:
        - Returns the mean of the target values.
        """
        # Start from the root of the tree
        tree = self
        # Traverse the tree until a leaf node is reached
        while 'value' not in tree:
            # Get the attribute and value used for splitting at the current node
            split_attribute = tree['split_attribute']
            split_val = tree['split_val']
            # Go to the left child if the record's value for the split attribute is less than or equal to the split value
            if record[split_attribute] <= split_val:
                tree = tree['left']
            # Otherwise, go to the right child
            else:
                tree = tree['right']
        # Return the value of the leaf node
        return tree['value']

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
                    prediction = DecisionTreeRegressor.Predict(OOB_tree, record)
                    predictions.append(prediction)

            if len(predictions) > 0:
                mean_prediction = np.mean(predictions)
                y.append(mean_prediction)
            else:
                # If the record is not out-of-bag (OOB), use all trees for prediction
                for i in range(self.num_trees):
                    tree = self.decision_trees[i]
                    predictions.append(DecisionTreeRegressor.Predict(tree, record))
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

