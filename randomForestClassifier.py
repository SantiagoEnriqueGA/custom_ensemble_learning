"""
This module contains the implementation of a Random Forest Classifier.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- RandomForestWithInfoGain: A class representing a Random Forest model that returns information gain for vis.
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

from decisionTreeClassifier import Utility, DecisionTree, DecisionTreeWithInfoGain

class RandomForest(object):
    """
    Random Forest classifier.

    Attributes:
        num_trees (int): The number of decision trees in the random forest.
        decision_trees (list): List of decision trees in the random forest.
        bootstraps_datasets (list): List of bootstrapped datasets for each tree.
        bootstraps_labels (list): List of true class labels corresponding to records in the bootstrapped datasets.
        max_depth (int): The maximum depth of each decision tree.

    Methods:
        __init__(self, num_trees, max_depth): Initializes the RandomForest object.
        _reset(self): Resets the RandomForest object.
        _bootstrapping(self, XX, n): Performs bootstrapping on the dataset.
        bootstrapping(self, XX): Initializes the bootstrapped datasets for each tree.
        fitting(self): Fits the decision trees to the bootstrapped datasets.
        voting(self, X): Performs voting to classify the input records.
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
        self.num_trees = num_trees      # Set the number of decision trees in the random forest
        self.max_depth = max_depth      # Set the maximum depth of each decision tree

        self.decision_trees = [DecisionTree(max_depth) for i in range(num_trees)]   # Initialize the decision trees
        
        self.bootstraps_datasets = []   # Initialize the list of bootstrapped datasets for each tree
        self.bootstraps_labels = []     # Initialize the list of true class labels corresponding to records in the bootstrapped datasets

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

        sample = [XX[i][:-1] for i in sample_indices]                       # Get the features of the selected samples
        labels = [XX[i][-1] for i in sample_indices]                        # Get the labels of the selected samples
        
        return (sample, labels)

    def bootstrapping(self, XX):
        """
        Initializes the bootstrapped datasets for each tree.

        Args:
            XX (list): The dataset.
        """
        for i in range(self.num_trees):                                 # For each decision tree
            data_sample, data_label = self._bootstrapping(XX, len(XX))  # Perform bootstrapping, using the entire dataset
            self.bootstraps_datasets.append(data_sample)                # Append the bootstrapped dataset, excluding the class labels
            self.bootstraps_labels.append(data_label)                   # Append the true class labels

    def fitting(self):
        """
        Fits the decision trees to the bootstrapped datasets.
        """
        for i in range(self.num_trees):                 # For each decision tree
            tree = self.decision_trees[i]               # Get the current tree        
            dataset = self.bootstraps_datasets[i]       # Get the bootstrapped dataset
            labels = self.bootstraps_labels[i]          # Get the true class labels
            
            self.decision_trees[i] = tree.learn(dataset, labels)    # Fit the tree to the bootstrapped dataset

    def voting(self, X):
        """
        Performs voting to classify the input records.

        Args:
            X (list): The input records.

        Returns:
            list: The predicted class labels for the input records.
        """
        y = []
        for record in X:    # For each record
            votes = []      
            for i, dataset in enumerate(self.bootstraps_datasets):  # For each bootstrapped dataset

                # Records not in the dataset are considered out-of-bag (OOB) records, which can be used for voting
                if record not in dataset:                           # If the record is not in the dataset
                    OOB_tree = self.decision_trees[i]               # Get the decision tree corresponding to the dataset
                    effective_vote = DecisionTree.classify(OOB_tree,record) # Classify the record using the decision tree
                    votes.append(effective_vote)                    # Append the classification to the votes list

            # Determine the majority vote
            if len(votes) > 0:                      # If there are votes
                counts = np.bincount(votes)         # Count the votes
                majority_vote = np.argmax(counts)   # Get the majority vote
                y.append(majority_vote)             # Append the majority vote to the list
            
            else:   # Can occur if the record is in all bootstrapped datasets
                y.append(np.random.choice([0, 1]))  # If there are no votes, randomly choose a class label

        return y

class RandomForestWithInfoGain(RandomForest):
    """
    A random forest classifier that uses information gain as the criterion for splitting.

    Parameters:
    - num_trees (int): The number of decision trees in the random forest.

    Attributes:
    - info_gains (list): A list to store the information gains of each decision tree.
    - decision_trees (list): A list of decision trees in the random forest.

    Methods:
    - fitting(): Fits the decision trees to the bootstrapped datasets.
    - display_info_gains(): Displays the information gains of each decision tree.
    - plot_info_gains_together(): Plots the information gains of all decision trees together.
    - plot_info_gains(): Plots the information gain of each decision tree separately.
    """

    def __init__(self, num_trees, max_depth):
        super().__init__(num_trees, max_depth)  # Initialize the RandomForest object
        self.info_gains = []                    # Initialize the list to store the information gains of each decision tree
        self.decision_trees = [DecisionTreeWithInfoGain(max_depth) for i in range(num_trees)]   # Initialize the decision trees

    def fitting(self):
        """
        Fits the decision trees to the bootstrapped datasets.
        """
        for i in range(self.num_trees):             # For each decision tree
            tree = self.decision_trees[i]           # Get the current tree
            dataset = self.bootstraps_datasets[i]   # Get the bootstrapped dataset
            labels = self.bootstraps_labels[i]      # Get the true class labels
            
            self.decision_trees[i] = tree.learn(dataset, labels)    # Fit the tree to the bootstrapped dataset
            self.info_gains.append(tree.info_gain)                  # Append the information gain to the list

    def display_info_gains(self):
        """
        Displays the information gains of each decision tree.
        """
        for i, info_gain in enumerate(self.info_gains): # For each decision tree
            print(f"Information gain of tree {i+1}:")   # Print the information gain of the tree
            
            for j, gain in enumerate(info_gain):        # For each split
                print(f"        split {j}: {gain}")     # Print the information gain of the split

    def plot_info_gains_together(self):
        """
        Plots the information gains of all decision trees together.
        """
        for i, info_gain in enumerate(self.info_gains):     # For each decision tree
            plt.plot(info_gain, label=f"Tree {i+1}")        # Plot the information gain
        plt.xlabel("Split")
        plt.ylabel("Information Gain")
        plt.title("Information Gain of Decision Trees")
        plt.legend()
        plt.show()

    def plot_info_gains(self):
        """
        Plots the information gain of each decision tree separately.
        """
        for i, info_gain in enumerate(self.info_gains):     # For each decision tree, plot the information gain
            plt.plot(info_gain)
            plt.xlabel("Split")
            plt.ylabel("Information Gain")
            plt.title(f"Information Gain of Decision Tree {i+1}")
            plt.show()

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

    def __init__(self, file_loc, display=False, forest_size=5, random_seed=0, max_depth=10):
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
        self.display = display          # Set the flag to display additional information about info gain
        
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

        if(self.display==False):    
            randomForest = RandomForest(self.forest_size,self.max_depth)                # If display is false, use the normal random forest
        else:
            randomForest = RandomForestWithInfoGain(self.forest_size, self.max_depth)   # If display is true, use the random forest with information gain

        print("creating the bootstrap datasets")
        randomForest.bootstrapping(self.XX)         # Creating the bootstrapped datasets

        print("fitting the forest")
        randomForest.fitting()                      # Fitting the decision trees to the bootstrapped datasets
        y_predicted = randomForest.voting(self.X)   # Voting to classify the input records

        results = [prediction == truth for prediction, truth in zip(y_predicted, self.y)]   # Comparing the predicted labels with the true labels

        accuracy = float(results.count(True)) / float(len(results)) # Calculating the accuracy
        
        # Displaying the results
        print("accuracy: %.4f" % accuracy)                          
        print("OOB estimate: %.4f" % (1 - accuracy))
        print("Execution time: " + str(datetime.now() - start))

        # Displaying additional information about info gain
        if(self.display==True):
            randomForest.display_info_gains()           # Display the information gains of each decision tree
            randomForest.plot_info_gains_together()     # Plot the information gains of all decision trees together
            randomForest.plot_info_gains()              # Plot the information gain of each decision tree separately
        
        return randomForest,accuracy



