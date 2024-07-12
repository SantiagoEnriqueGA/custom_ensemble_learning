import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import Custom Classes
import gradientBoostedRegressor as gbr
import randomForestRegressor as rfg
import randomForestClassifier as rfc
import dataPrep as dp


def randomForestDiabetes():
    """
    Runs Random Forest Classifier on Pima Indians Diabetes dataset.
    Visualizes the accuracy for different forest sizes and maximum depths.
    """
    print("\n\nRandom Forest Classifier on Pima Indians Diabetes dataset\n")

    # Source file location
    file_orig = "data/pima-indians-diabetes.csv"

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=2, cols_to_encode=[9, 10])

    # Define the range of forest sizes and maximum depths to test
    forest_sizes = [10, 20, 50, 100]
    max_depths = [10, 15, 20, 25]

    # Store results
    results = np.zeros((len(forest_sizes), len(max_depths)))

    # Loop over different forest sizes and maximum depths
    for i, forest_size in enumerate(forest_sizes):
        for j, max_depth in enumerate(max_depths):
            # Initialize random forest object
            rfDiab = rfc.runRandomForest(file_loc, False, forest_size=forest_size, random_seed=0, max_depth=max_depth)
            
            # Train random forest model and get accuracy
            randomForest, accuracy = rfDiab.run()
            results[i, j] = accuracy

            print(f"Forest Size: {forest_size}, Max Depth: {max_depth}, Accuracy: {accuracy}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    for i, forest_size in enumerate(forest_sizes):
        plt.plot(max_depths, results[i, :], label=f'Forest Size: {forest_size}')
    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Classifier Accuracy for Different Forest Sizes and Max Depths')
    plt.legend()
    plt.grid(True)
    plt.show()

def randomForestBreastCancer():
    """
    Runs Random Forest Classifier on Wisconsin Breast Prognostic dataset.
    Visualizes the accuracy for different forest sizes and maximum depths.
    """
    print("\n\nRandom Forest Classifier on Wisconsin Breast Prognostic dataset\n")

    # Source file location
    file_orig = "data/Wisconsin_breast_prognostic.csv"
    
    # File already formatted
    file_loc = file_orig

    # Define the range of forest sizes and maximum depths to test
    forest_sizes = [1, 10, 20, 50, 100]
    max_depths = [2, 5, 10, 15]

    # Store results
    results = np.zeros((len(forest_sizes), len(max_depths)))

    # Loop over different forest sizes and maximum depths
    for i, forest_size in enumerate(forest_sizes):
        for j, max_depth in enumerate(max_depths):
            # Initialize random forest object
            rfObjBreastCancer = rfc.runRandomForest(file_loc, False, forest_size=forest_size, random_seed=0, max_depth=max_depth)
            
            # Train random forest model and get accuracy
            randomForest, accuracy = rfObjBreastCancer.run()
            results[i, j] = accuracy

            print(f"Forest Size: {forest_size}, Max Depth: {max_depth}, Accuracy: {accuracy}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    for i, forest_size in enumerate(forest_sizes):
        plt.plot(max_depths, results[i, :], label=f'Forest Size: {forest_size}')
    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Classifier Accuracy for Different Forest Sizes and Max Depths (Breast Cancer Dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()

def randomForestCarsReg():
    """
    Runs Random Forest Regressor on Cars.com dataset.
    Visualizes the performance metrics for different forest sizes and maximum depths.
    """
    print("\n\nRandom Forest Regressor on Cars.com dataset\n")

    df = pd.read_csv("data/output_May-06-2024_cleaned.csv")
    df = df[['Miles', 'Stock', 'Year', 'Sub_Model', 'Price']]
    df.to_csv("data/carsDotCom.csv", index=False)

    # Source file location
    file_orig = "data/carsDotCom.csv"

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=4, cols_to_encode=[1, 2, 3])

    # Define the range of forest sizes and maximum depths to test
    forest_sizes = [3, 10, 20, 50]
    max_depths = [5, 10, 15, 20]

    # Store results
    results = {metric: np.zeros((len(forest_sizes), len(max_depths))) for metric in ['MSE', 'R^2', 'MAPE', 'MAE', 'RMSE']}

    # Loop over different forest sizes and maximum depths
    for i, forest_size in enumerate(forest_sizes):
        for j, max_depth in enumerate(max_depths):
            # Initialize random forest object
            rfObj = rfg.runRandomForest(file_loc, forest_size=forest_size, random_seed=0, max_depth=max_depth)
            
            # Train random forest model and get stats
            randomForest, stats = rfObj.run()
            for metric in results:
                results[metric][i, j] = stats[metric]

            print(f"Forest Size: {forest_size}, Max Depth: {max_depth}, Stats: {stats}")

    # Plot the results for each metric
    for metric in results:
        plt.figure(figsize=(12, 6))
        for i, forest_size in enumerate(forest_sizes):
            plt.plot(max_depths, results[metric][i, :], label=f'Forest Size: {forest_size}')
        
        plt.xlabel('Max Depth')
        plt.ylabel(metric)
        plt.title(f'Random Forest Regressor {metric} for Different Forest Sizes and Max Depths (Cars.com Dataset)')
        plt.legend()
        plt.grid(True)
        plt.show()

def gbtrCarsReg():
    """
    Runs Gradient Boosted Regressor on the Cars.com dataset.
    Visualizes the performance metrics for different numbers of trees and maximum depths.
    """
    print("\n\nGradient Boosted Regressor on Cars.com dataset\n")

    # Source file location
    file_orig = "data/carsDotCom.csv"

    # Prepare and format data
    df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=4, cols_to_encode=[1, 2, 3])

    # Define the range of numbers of trees and maximum depths to test
    num_trees_list = [10, 20, 50, 100]
    max_depths = [5, 10, 15, 20]

    # Store results
    results = {metric: np.zeros((len(num_trees_list), len(max_depths))) for metric in ['MSE', 'R^2', 'MAPE', 'MAE', 'RMSE']}
    mean_absolute_residuals = {}

    # Loop over different numbers of trees and maximum depths
    for i, num_trees in enumerate(num_trees_list):
        for j, max_depth in enumerate(max_depths):
            # Initialize GBDT object
            gbdtDiab = gbr.gradientBoostedRegressor(file_loc, num_trees=num_trees, random_seed=0, max_depth=max_depth)
            
            # Train GBDT model
            gbdtDiab.fit(stats=True)

            # Predict target values
            predictions = gbdtDiab.predict()

            # Get stats
            stats = gbdtDiab.get_stats(predictions)
            for metric in results:
                results[metric][i, j] = stats[metric]
            mean_absolute_residuals[(num_trees, max_depth)] = stats['Mean_Absolute_Residuals']

            print(f"Num Trees: {num_trees}, Max Depth: {max_depth}, Stats: {stats}")

    # Plot the results for each metric
    for metric in results:
        plt.figure(figsize=(12, 6))
        for i, num_trees in enumerate(num_trees_list):
            plt.plot(max_depths, results[metric][i, :], label=f'Num Trees: {num_trees}')
        
        plt.xlabel('Max Depth')
        plt.ylabel(metric)
        plt.title(f'Gradient Boosted Regressor {metric} for Different Numbers of Trees and Max Depths (Cars.com Dataset)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot Mean Absolute Residuals
    plt.figure(figsize=(12, 6))
    for (num_trees, max_depth), residuals in mean_absolute_residuals.items():
        plt.plot(range(len(residuals)), residuals, label=f'Num Trees: {num_trees}, Max Depth: {max_depth}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Residual')
    plt.title('Gradient Boosted Regressor Mean Absolute Residuals for Different Numbers of Trees and Max Depths (Cars.com Dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()





# randomForestDiabetes()
randomForestBreastCancer()
# randomForestCarsReg()
# gbtrCarsReg()


