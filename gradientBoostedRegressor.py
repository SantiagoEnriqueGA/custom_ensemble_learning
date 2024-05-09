
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
        # self.reset()

        self.random_seed = random_seed
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.X = list()
        self.y = list()
        self.XX = list()  # Contains both data features and data labels
        self.numerical_cols = 0

        self.utility = Utility()
        self.trees = [DecisionTreeRegressor(self.max_depth) for i in range(self.num_trees)]

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
        self.random_seed = 0
        self.num_trees = 10
        self.max_depth = 10
        self.X = list()
        self.y = list()
        self.XX = list()
        self.numerical_cols = 0


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

        # Check if the input data X and target values y are not empty
        if not self.X or not self.y:
            raise ValueError("Input data X and target values y cannot be empty.")
        
        # Initialize residuals as a numpy array of target values y
        residuals = np.array(self.y)

        # Loop over the number of trees in the ensemble
        for i in range(self.num_trees):
            # Update the tree by learning from the input data X and residuals
            tree = self.trees[i]
            self.trees[i] = tree.learn(self.X, residuals)

            # Predict the target values for the input data X using the tree
            predictions = np.array([DecisionTreeRegressor.predict(self.trees[i], record) for record in self.X])
        
            # Update the residuals by subtracting the predictions from the target values
            residuals = residuals - predictions

            # If True, print the number of the trained tree and the total residuals
            if stats:
                print(f"Tree {i+1} trained. Mean Absolute Residuals: {abs(residuals).mean()}")

    def predict(self):
        """
        Predicts the target values for the input data using the gradient boosted decision tree regressor.

        Returns:
            predictions (numpy.ndarray): An array of predicted target values for the input data.
        """

        # Initialize an array of zeros with the same length as the input data
        predictions = np.zeros(len(self.X))

        # Loop over each tree in the ensemble
        for i in range(self.num_trees):
            # Initialize an array of zeros for the predictions of current tree
            oneTree_predictions = np.zeros(len(self.X))

            # For each data point in the input data, predict the target value using the current tree
            for j in range(len(self.X)):
                oneTree_predictions[j] += DecisionTreeRegressor.predict(self.trees[i], self.X[j])

            # Add the predictions of the current tree to the overall predictions
            predictions += oneTree_predictions
            
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
        # Mean Squared Error (MSE)
        mse = np.mean((np.array(y_predicted) - np.array(self.y)) ** 2)

        # R^2 Score
        ssr = np.sum((np.array(y_predicted) - np.array(self.y)) ** 2)
        sst = np.sum((np.array(self.y) - np.mean(self.y)) ** 2)
        r2 = 1 - (ssr / sst)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((np.array(self.y) - np.array(y_predicted)) / np.array(self.y))) * 100

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(np.array(self.y) - np.array(y_predicted)))

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((np.array(y_predicted) - np.array(self.y)) ** 2))

        return {
            "MSE": mse,
            "R^2": r2,
            "MAPE": mape,
            "MAE": mae,
            "RMSE": rmse
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