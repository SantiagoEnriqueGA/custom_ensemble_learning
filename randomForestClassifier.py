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
        user(self): Returns the user's GTUsername.
    """

    # Initialize class variables
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
        # Set the number of trees
        self.num_trees = num_trees
        # Set the maximum depth for each tree
        self.max_depth = max_depth
        # Create the decision trees
        self.decision_trees = [DecisionTree(max_depth) for i in range(num_trees)]
        # Initialize the bootstraps datasets and labels
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
        # Select samples from the dataset
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
        # Perform bootstrapping for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        """
        Fits the decision trees to the bootstrapped datasets.
        """
        # Fit each decision tree to the bootstrapped datasets
        for i in range(self.num_trees):
            tree = self.decision_trees[i]
            dataset = self.bootstraps_datasets[i]
            labels = self.bootstraps_labels[i]
            self.decision_trees[i] = tree.learn(dataset, labels)

    def voting(self, X):
        """
        Performs voting to classify the input records.

        Args:
            X (list): The input records.

        Returns:
            list: The predicted class labels for the input records.
        """
        y = []

        # Perform voting for each record
        for record in X:
            votes = []

            for i, dataset in enumerate(self.bootstraps_datasets):
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = DecisionTree.classify(OOB_tree,record)
                    votes.append(effective_vote)

            # Determine the majority vote
            if len(votes) > 0:
                counts = np.bincount(votes)
                majority_vote = np.argmax(counts)
                y.append(majority_vote)
            else:
                y.append(np.random.choice([0, 1]))

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
        super().__init__(num_trees, max_depth)
        self.info_gains = []
        self.decision_trees = [DecisionTreeWithInfoGain(max_depth) for i in range(num_trees)]

    def fitting(self):
        """
        Fits the decision trees to the bootstrapped datasets.
        """
        for i in range(self.num_trees):
            tree = self.decision_trees[i]
            dataset = self.bootstraps_datasets[i]
            labels = self.bootstraps_labels[i]
            # print("Fitting tree: ",i+1)
            self.decision_trees[i] = tree.learn(dataset, labels)
            self.info_gains.append(tree.info_gain)

    def display_info_gains(self):
        """
        Displays the information gains of each decision tree.
        """
        for i, info_gain in enumerate(self.info_gains):
            print(f"Information gain of tree {i+1}:")
            for j, gain in enumerate(info_gain):
                print(f"        split {j}: {gain}")

    def plot_info_gains_together(self):
        """
        Plots the information gains of all decision trees together.
        """
        for i, info_gain in enumerate(self.info_gains):
            plt.plot(info_gain, label=f"Tree {i+1}")
        plt.xlabel("Split")
        plt.ylabel("Information Gain")
        plt.title("Information Gain of Decision Trees")
        plt.legend()
        plt.show()

    def plot_info_gains(self):
        """
        Plots the information gain of each decision tree separately.
        """
        for i, info_gain in enumerate(self.info_gains):
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

    random_seed = 0
    forest_size = 10
    max_depth = 10
    display = False
    X = list()
    y = list()
    XX = list()  # Contains both data features and data labels
    numerical_cols = 0

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
        self.reset()

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.forest_size = forest_size
        self.max_depth = max_depth
        self.display = display
        
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
        if(self.display==False):
            randomForest = RandomForest(self.forest_size,self.max_depth)
        else:
            randomForest = RandomForestWithInfoGain(self.forest_size, self.max_depth)

        # Creating the bootstrapping datasets
        print("creating the bootstrap datasets")
        randomForest.bootstrapping(self.XX)

        # Building trees in the forest
        print("fitting the forest")
        randomForest.fitting()

        # Calculating an unbiased error estimation of the random forest based on out-of-bag (OOB) error estimate.
        y_predicted = randomForest.voting(self.X)

        # Comparing predicted and true labels
        results = [prediction == truth for prediction, truth in zip(y_predicted, self.y)]

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        print("accuracy: %.4f" % accuracy)
        print("OOB estimate: %.4f" % (1 - accuracy))

        # End time
        print("Execution time: " + str(datetime.now() - start))

        # Displaying additional information about info gain
        if(self.display==True):
            randomForest.display_info_gains()
            randomForest.plot_info_gains_together()
            randomForest.plot_info_gains()
        return randomForest,accuracy



