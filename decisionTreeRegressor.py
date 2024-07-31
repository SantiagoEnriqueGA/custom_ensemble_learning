"""
This module contains the implementation of a Decision Tree Regressor.

The module includes the following classes:
- Utility: A utility class for computing variance, partitioning classes, and calculating information gain.
- DecisionTreeRegressor: A class representing a decision tree for regression.
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
    Utility class for computing variance, partitioning classes, and calculating information gain.
    """

    def calculate_variance(self, y):
        """
        Calculate the variance of a dataset.
        Variance is used as the measure of impurity in the case of regression.
        """
        if len(y) == 0:     # If the dataset is empty
            return 0        # Return 0 variance
        
        mean = sum(y) / len(y)                                  # Calculate the mean of the dataset
        variance = sum((yi - mean) ** 2 for yi in y) / len(y)   # Calculate the variance using the mean
        
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

        if X.ndim == 1:             # If X is a 1D array
            X = X.reshape(-1, 1)    # Convert to a 2D array with one column

        # Use NumPy boolean indexing for partitioning
        # X_left  contains rows where the split attribute is less than or equal to the split value
        # X_right contains rows where the split attribute is greater than the split value
        # y_left  contains target labels corresponding to X_left
        # y_right contains target labels corresponding to X_right
        X_left = X[X[:, split_attribute] <= split_val]
        X_right = X[X[:, split_attribute] > split_val]
        y_left = y[X[:, split_attribute] <= split_val]
        y_right = y[X[:, split_attribute] > split_val]

        return X_left, X_right, y_left, y_right     # Return the partitioned subsets

    def information_gain(self, previous_y, current_y):
        """
        Calculate the information gain from a split by subtracting the variance of
        child nodes from the variance of the parent node.
        """
        parent_variance = self.calculate_variance(previous_y)                                           # Calculate the variance of the parent node
        child_variance = sum(self.calculate_variance(y) * len(y) for y in current_y) / len(previous_y)  # Calculate the variance of the child nodes

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
        num_features = int(np.sqrt(X.shape[1]))                                                 # Square root of total attributes
        selected_attributes = np.random.choice(X.shape[1], size=num_features, replace=False)    # Randomly select attributes

        # Initialize the best information gain to negative infinity, others to None
        best_info_gain = float('-inf')
        best_split_attribute = None
        best_split_val = None
        best_X_left, best_X_right, best_y_left, best_y_right = None, None, None, None

        for split_attribute in selected_attributes:     # For each selected attribute
            values = np.unique(X[:, split_attribute])   # Get unique values of the attribute
            
            for split_val in values:                    # For each unique value
                X_left, X_right, y_left, y_right = self.partition_classes(X, y, split_attribute, split_val)     # Partition the data
                
                info_gain = self.information_gain(y, [y_left, y_right])     # Calculate information gain
                
                if info_gain > best_info_gain:              # If the information gain is better than the current best
                    best_info_gain = info_gain              # Update the best information gain
                    best_split_attribute = split_attribute  # Update the best split attribute
                    best_split_val = split_val              # Update the best split value
                    best_X_left, best_X_right, best_y_left, best_y_right = X_left, X_right, y_left, y_right  # Update the best subsets

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
        self.tree = {}              # Initialize an empty dictionary to represent the decision tree
        self.max_depth = max_depth  # Set the maximum depth of the tree

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
        y = y.tolist() if isinstance(y, np.ndarray) else y  # Convert y to a list if it is a NumPy array
        
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y, dtype=int)
        
        if len(set(y)) == 1:        # If the node is pure (all labels are the same)
            return {'value': y[0]}  # Return the label as the value of the leaf node

        if depth >= self.max_depth:         # If the maximum depth is reached
            return {'value': np.mean(y)}    # Return the mean of the target values as the value of the leaf node

        # Get the best split using utility functions
        best_split = Utility().best_split(X, y)
        split_attribute = best_split['split_attribute']
        split_val = best_split['split_val']
        X_left = best_split['X_left']
        X_right = best_split['X_right']
        y_left = best_split['y_left']
        y_right = best_split['y_right']

        if best_split['info_gain'] == 0:    # If the information gain is 0
            return {'value': np.mean(y)}    # Return the mean of the target values as the value of the leaf node

        # Recursively build the left and right subtrees
        par_node = {'split_attribute': split_attribute, 'split_val': split_val} # Set the split attribute and value
        par_node['left'] = self.learn(X_left, y_left, depth=depth + 1)          # Build the left subtree
        par_node['right'] = self.learn(X_right, y_right, depth=depth + 1)       # Build the right subtree

        return par_node

    def predict(self, record):
        """
        Predicts a given record using the decision tree.

        Parameters:
        - record: A dictionary representing the record to be classified.

        Returns:
        - Returns the mean of the target values.
        """
        tree = self     # Get the decision tree
        
        while 'value' not in tree:                      # While the current node is not a leaf node
            split_attribute = tree['split_attribute']   # Get the split attribute
            split_val = tree['split_val']               # Get the split value
            
            if record[split_attribute] <= split_val:    # If the record's attribute value is less than or equal to the split value
                tree = tree['left']                     # Go to the left child
            
            else:                                       # If the record's attribute value is greater than the split value
                tree = tree['right']                    # Go to the right child
        
        return tree['value']        # Return the value of the leaf node