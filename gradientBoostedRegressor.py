
from decisionTreeRegressor import Utility, DecisionTreeRegressor
import dataPrep as dp

import numpy as np
import csv
import ast

class gradientBoostedRegressor(object):
    """
    A class to represent a Gradient Boosted Decision Tree Regressor.

    Attributes:
        random_seed (int): The random seed for the random number generator.
        num_trees (int): The number of decision trees in the ensemble.
        max_depth (int): The maximum depth of each decision tree.
        display (bool): A flag to display the decision tree.
        X (list): A list of input data features.
        y (list): A list of target values.
        XX (list): A list of input data features and target values.
        numerical_cols (set): A set of indices of numeric attributes (columns).
    
    Methods:
        __init__(file_loc, num_trees=5, random_seed=0, max_depth=10): Initializes the GBDT object.
        reset(): Resets the GBDT object.
        fit(): Fits the GBDT model to the training data.
        predict(): Predicts the target values for the input data.
        get_stats(y_predicted): Calculates various evaluation metrics for the predicted target values.
    """

    def __init__(self, file_loc: str, num_trees: int = 5, random_seed: int = 0, max_depth: int = 10):
        self.random_seed = random_seed  # Set the random seed for reproducibility
        self.num_trees = num_trees      # Set the number of trees in the ensemble
        self.max_depth = max_depth      # Set the maximum depth of each tree

        self.X = list()                 # Initialize the list of input data features
        self.y = list()                 # Initialize the list of target values
        self.XX = list()                # Initialize the list of input data features and target values
        
        self.numerical_cols = 0             # Initialize the set of indices of numeric attributes (columns)
        self.mean_absolute_residuals = []   # Initialize the list of Mean Absolute Residuals for each tree

        self.utility = Utility()            # Initialize the Utility object
        self.trees = [DecisionTreeRegressor(self.max_depth) for i in range(self.num_trees)] # Initialize the list of decision trees
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
        # Reset the GBDT object
        self.random_seed = 0
        self.num_trees = 10
        self.max_depth = 10
        self.X = list()
        self.y = list()
        self.XX = list()
        self.numerical_cols = 0
        self.mean_absolute_residuals = []


    def fit(self, stats=False):
        """
        Fits the gradient boosted decision tree regressor to the training data.

        This method trains the ensemble of decision trees by iteratively fitting each tree to the residuals
        of the previous iteration. The residuals are updated after each iteration by subtracting the predictions
        made by the current tree from the target values.

        Args:
            stats (bool): A flag to decide whether to return stats or not. Default is False.

        Returns:
            None
        """
        if not self.X or not self.y:    # If the input data X or target values y are empty
            raise ValueError("Input data X and target values y cannot be empty.")
        
        residuals = np.array(self.y)    # Initialize the residuals with the target values

        for i in range(self.num_trees):     # Loop over the number of trees in the ensemble
            tree = self.trees[i]                            # Get the current tree
            self.trees[i] = tree.learn(self.X, residuals)   # Fit the tree to the residuals

            predictions = np.array([DecisionTreeRegressor.predict(self.trees[i], record) for record in self.X]) # Predict the target values using the current tree
        
            residuals = residuals - predictions     # Update the residuals by subtracting the predictions from the target values

            mean_absolute_residual = np.mean(np.abs(residuals))         # Calculate the mean absolute residuals
            self.mean_absolute_residuals.append(mean_absolute_residual) # Append the mean absolute residuals to the list

            if stats:   # If stats is True, print the mean absolute residuals
                print(f"Tree {i+1} trained. Mean Absolute Residuals: {mean_absolute_residual}")

    def predict(self):
        """
        Predicts the target values for the input data using the gradient boosted decision tree regressor.

        Returns:
            predictions (numpy.ndarray): An array of predicted target values for the input data.
        """
        predictions = np.zeros(len(self.X))     # Initialize an array of zeros for the predictions

        for i in range(self.num_trees):                     # Loop over the number of trees in the ensemble
            oneTree_predictions = np.zeros(len(self.X))     # Initialize an array of zeros for the predictions of the current tree

            for j in range(len(self.X)):                    # Loop over the indices of the input data
                oneTree_predictions[j] += DecisionTreeRegressor.predict(self.trees[i], self.X[j])   # Predict the target value for the current input data

            predictions += oneTree_predictions              # Add the predictions of the current tree to the overall predictions
            
        return predictions
    
    def get_stats(self, y_predicted):
        """
        Calculates various evaluation metrics for the predicted target values.

        Args:
            y_predicted (numpy.ndarray): An array of predicted target values.

        Returns:
            dict: A dictionary containing the evaluation metrics.
                - MSE (float): Mean Squared Error
                - R^2 (float): R-squared Score
                - MAPE (float): Mean Absolute Percentage Error
                - MAE (float): Mean Absolute Error
                - RMSE (float): Root Mean Squared Error
        """
        mse = np.mean((np.array(y_predicted) - np.array(self.y)) ** 2)  # Mean Squared Error (MSE): (y - y')^2

        ssr = np.sum((np.array(y_predicted) - np.array(self.y)) ** 2)   # Sum of Squared Residuals (SSR): (y - y')^2
        sst = np.sum((np.array(self.y) - np.mean(self.y)) ** 2)         # Total Sum of Squares (SST): (y - mean(y))^2
        r2 = 1 - (ssr / sst)                                            # R-squared Score (R^2): 1 - (SSR / SST)

        mape = np.mean(np.abs((np.array(self.y) - np.array(y_predicted)) / np.array(self.y))) * 100     # Mean Absolute Percentage Error (MAPE): (|y - y'| / y) * 100

        mae = np.mean(np.abs(np.array(self.y) - np.array(y_predicted))) # Mean Absolute Error (MAE): |y - y'|

        rmse = np.sqrt(np.mean((np.array(y_predicted) - np.array(self.y)) ** 2))    # Root Mean Squared Error (RMSE): sqrt((y - y')^2)

        # Return the evaluation metrics
        return {
            "MSE": mse,
            "R^2": r2,
            "MAPE": mape,
            "MAE": mae,
            "RMSE": rmse,
            "Mean_Absolute_Residuals": self.mean_absolute_residuals
        }

  
# Example Usage:
def run():
    """
    Runs Gradient Boosted Decision Trees on the given dataset.
    """
    # Source file location
    file_orig = "data/carsDotCom.csv"

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=4, cols_to_encode=[1,2,3])

    # Initialize GBDT object
    gbdtDiab = gradientBoostedRegressor(file_loc, num_trees=10, random_seed=0, max_depth=3)

    # Train GBDT model
    gbdtDiab.fit(stats=True)

    # Predict target values
    predictions = gbdtDiab.predict()

    # Get stats
    stats = gbdtDiab.get_stats(predictions)
    print(stats)
if __name__ == "__main__":
    run()