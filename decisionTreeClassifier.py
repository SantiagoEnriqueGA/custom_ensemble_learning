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
        counts = np.bincount(class_y)                               # Count the occurrences of each class
        probabilities = counts / float(len(class_y))                # Calculate the probabilities
        probabilities = probabilities[probabilities > 0]            # Remove zero probabilities
        entropy = -np.sum(probabilities * np.log2(probabilities))   # Calculate entropy, Σ(-p(x) * log2(p(x)))
        
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
        - X_left  (array-like): The subset of input features where the split attribute is less than or equal to the split value.
        - X_right (array-like): The subset of input features where the split attribute is greater than the split value.
        - y_left  (array-like): The subset of target labels corresponding to X_left.
        - y_right (array-like): The subset of target labels corresponding to X_right.
        """
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:             # If X has only one feature
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
        Calculates the information gain between the previous and current values of y.

        Parameters:
        - previous_y (array-like): The previous values of y.
        - current_y (array-like): The current values of y.

        Returns:
        - float: The information gain between the previous and current values of y.
        """
        entropy_prev = self.entropy(previous_y) # Compute the entropy of the previous y values
        total_count = len(previous_y)           # Get the total count of previous y values
        
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
        num_features = int(np.sqrt(X.shape[1]))                                                 # Square root of total attributes
        selected_attributes = np.random.choice(X.shape[1], size=num_features, replace=False)    # Randomly select attributes

        # Initialize the best information gain to negative infinity, others to None
        best_info_gain = float('-inf')
        best_split_attribute = None
        best_split_val = None
        best_X_left, best_X_right, best_y_left, best_y_right = None, None, None, None

        for split_attribute in selected_attributes:     # For each selected attribute
            values = np.unique(X[:, split_attribute])   # Get the unique values of the attribute
            
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
        self.tree = {}              # Initialize the tree as an empty dictionary
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
        y = y.tolist() if isinstance(y, np.ndarray) else y  # Convert y to a Python list
        
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y, dtype=int)
        
        if len(set(y)) == 1:        # If the node is pure (all labels are the same)
            return {'label': y[0]}  # Return the label of the node

        if depth >= self.max_depth:                     # If maximum depth is reached
            return {'label': np.argmax(np.bincount(y))} # Return the most common label

        # Get the best split using utility function
        best_split = Utility().best_split(X, y)
        split_attribute = best_split['split_attribute']
        split_val = best_split['split_val']
        X_left = best_split['X_left']
        X_right = best_split['X_right']
        y_left = best_split['y_left']
        y_right = best_split['y_right']

        if best_split['info_gain'] == 0:                        # If there is no further information gain
            return {'label': max(set(y), key=list(y).count)}    # Return the most common label

        # Recursively build the left and right subtrees
        par_node = {'split_attribute': split_attribute, 'split_val': split_val} # Set the split attribute and value
        par_node['left'] = self.learn(X_left, y_left, depth=depth + 1)          # Build the left subtree
        par_node['right'] = self.learn(X_right, y_right, depth=depth + 1)       # Build the right subtree

        return par_node 

    def classify(self, record):
        """
        Classifies a given record using the decision tree.

        Parameters:
        - record: A dictionary representing the record to be classified.

        Returns:
        - The label assigned to the record based on the decision tree.
        """
        tree = self     # Get the decision tree

        while 'label' not in tree:                      # While the current node is not a leaf node
            split_attribute = tree['split_attribute']   # Get the split attribute
            split_val = tree['split_val']               # Get the split value
            
            if record[split_attribute] <= split_val:    # If the record's attribute value is less than or equal to the split value
                tree = tree['left']                     # Go to the left child
            
            else:                                       # If the record's attribute value is greater than the split value
                tree = tree['right']                    # Go to the right child
        
        return tree['label']    # Return the label assigned to the record
    

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
        super().__init__(max_depth)     # Initialize the DecisionTree class
        self.info_gain = []             # Initialize the information gain list
    
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
        y = y.tolist() if isinstance(y, np.ndarray) else y  # Convert y to a Python list
        
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y, dtype=int)
        
        if len(set(y)) == 1:        # If the node is pure (all labels are the same)
            return {'label': y[0]}  # Return the label of the node

        if depth >= self.max_depth:                     # If maximum depth is reached
            return {'label': np.argmax(np.bincount(y))} # Return the most common label

        # Get the best split using utility functions
        best_split = Utility().best_split(X, y)
        split_attribute = best_split['split_attribute']
        split_val = best_split['split_val']
        X_left = best_split['X_left']
        X_right = best_split['X_right']
        y_left = best_split['y_left']
        y_right = best_split['y_right']

        self.info_gain.append(best_split['info_gain'])  # Append the information gain to the list

        if best_split['info_gain'] == 0:                        # If there is no further information gain
            return {'label': max(set(y), key=list(y).count)}    # Return the most common label

        # Recursively build the left and right subtrees
        par_node = {'split_attribute': split_attribute, 'split_val': split_val} # Set the split attribute and value
        par_node['left'] = self.learn(X_left, y_left, depth=depth + 1)          # Build the left subtree
        par_node['right'] = self.learn(X_right, y_right, depth=depth + 1)       # Build the right subtree

        return par_node