"""
This module contains the implementation of a Decision Tree Classifier.

The module includes the following classes:
- Utility: A utility class for computing entropy, partitioning classes, and calculating information gain.
- DecisionTree: A class representing a decision tree for classification.
- DecisionTreeWithInfoGain: A class representing a decision tree for classification with information gain.
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

    def entropy(self, class_y):
        """
        Computes the entropy for a given class.

        Parameters:
        - class_y (array-like): The class labels.

        Returns:
        - float: The entropy value.
        """
        # Calculate the count of each class
        counts = np.bincount(class_y)
        # Probabilities of each class
        probabilities = counts / float(len(class_y))
        # Ignore zero probabilities
        probabilities = probabilities[probabilities > 0]
        # Calculate entropy -> - Σ (p_i * log2(p_i))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy


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
        Calculates the information gain between the previous and current values of y.

        Parameters:
        - previous_y (array-like): The previous values of y.
        - current_y (array-like): The current values of y.

        Returns:
        - float: The information gain between the previous and current values of y.
        """
        # Compute the entropy of the previous y values
        entropy_prev = self.entropy(previous_y)
        
        # Get the total count of previous y values
        total_count = len(previous_y)
        
        # Compute the weighted entropy of the current y values
        # For each subset in current_y, calculate its entropy and multiply it by the proportion of the subset in the total count
        entropy_current = np.sum([(len(subset) / total_count) * self.entropy(subset) for subset in current_y])
        
        # Information gain is the difference between the entropy of the previous y values and the weighted entropy of the current y values
        info_gain = entropy_prev - entropy_current
        
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

class DecisionTree(object):
    """
    A class representing a decision tree.

    Parameters:
    - max_depth (int): The maximum depth of the decision tree.

    Methods:
    - learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
    - classify(record): Classifies a record using the decision tree.

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
        
        # Check if the node is pure (all labels are the same)
        if len(set(y)) == 1:
            return {'label': y[0]}

        # Check if maximum depth is reached
        if depth >= self.max_depth:
            return {'label': np.argmax(np.bincount(y))}

        # Get the best split using utility functions
        best_split = Utility().best_split(X, y)
        split_attribute = best_split['split_attribute']
        split_val = best_split['split_val']
        X_left = best_split['X_left']
        X_right = best_split['X_right']
        y_left = best_split['y_left']
        y_right = best_split['y_right']

        # Check if there is no further information gain
        if best_split['info_gain'] == 0:
            return {'label': max(set(y), key=list(y).count)}

        # Recursively build the left and right subtrees
        par_node = {'split_attribute': split_attribute, 'split_val': split_val}
        par_node['left'] = self.learn(X_left, y_left, depth=depth + 1)
        par_node['right'] = self.learn(X_right, y_right, depth=depth + 1)

        return par_node

    def classify(self, record):
        """
        Classifies a given record using the decision tree.

        Parameters:
        - record: A dictionary representing the record to be classified.

        Returns:
        - The label assigned to the record based on the decision tree.
        """
        # Start from the root of the tree
        tree = self
        # Traverse the tree until a leaf node is reached
        while 'label' not in tree:
            # Get the attribute and value used for splitting at the current node
            split_attribute = tree['split_attribute']
            split_val = tree['split_val']
            # Go to the left child if the record's value for the split attribute is less than or equal to the split value
            if record[split_attribute] <= split_val:
                tree = tree['left']
            # Otherwise, go to the right child
            else:
                tree = tree['right']
        # Return the label of the leaf node
        return tree['label']
    
class DecisionTreeWithInfoGain(DecisionTree):
    """
    A class representing a decision tree.

    Parameters:
    - max_depth (int): The maximum depth of the decision tree.

    Methods:
    - learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
    - classify(record): Classifies a record using the decision tree.

    """
    
    def __init__(self, max_depth=None):
        super().__init__(max_depth)
        self.info_gain = []
    
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
        
        # Check if the node is pure (all labels are the same)
        if len(set(y)) == 1:
            return {'label': y[0]}

        # Check if maximum depth is reached
        if depth >= self.max_depth:
            return {'label': np.argmax(np.bincount(y))}

        # Get the best split using utility functions
        best_split = Utility().best_split(X, y)
        split_attribute = best_split['split_attribute']
        split_val = best_split['split_val']
        X_left = best_split['X_left']
        X_right = best_split['X_right']
        y_left = best_split['y_left']
        y_right = best_split['y_right']

        # print("Info Gain: ",best_split['info_gain'])
        self.info_gain.append(best_split['info_gain'])

        # Check if there is no further information gain
        if best_split['info_gain'] == 0:
            return {'label': max(set(y), key=list(y).count)}

        # Recursively build the left and right subtrees
        par_node = {'split_attribute': split_attribute, 'split_val': split_val}
        par_node['left'] = self.learn(X_left, y_left, depth=depth + 1)
        par_node['right'] = self.learn(X_right, y_right, depth=depth + 1)

        return par_node