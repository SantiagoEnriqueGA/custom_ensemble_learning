"""
Implementation of a random forest classifier. The performance of the classifier is evaluated via the out-of-bag (OOB) error estimate. 
Used on dataset wisconsin_breast_prognostic.csv :
	Features (Attributes) were computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  
	They describe characteristics of the cell nuclei present in the image.   
	Each row describes one patient (a data point, or data record) and each row includes 31 columns.  
	The first 30 columns are attributes.  
	The 31st (last column) is the label.  
	The value one and zero indicates whether the cancer is malignant or benign.  
Model performs binary classification on the dataset to determine if a particular cancer is benign or malignant. 
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

    def user(self):
        """
        Returns the user's GTUsername.

        Returns:
            str: The user's GTUsername.
        """
        return 'sangarita3'

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

        # printing the name
        print("__Name: " + randomForest.user()+"__")

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



