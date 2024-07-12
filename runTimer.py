import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# Import Custom Classes
import randomForestRegressor as rfg
import randomForestClassifier as rfc
import randomForestClassifierPar as rfcp
import dataPrep as dp

import multiprocessing

def runRandomForest(file_loc, multiprocessing_enabled, forest_size, max_depth=25):
    if multiprocessing_enabled:
        return rfcp.runRandomForest(file_loc, False, forest_size=forest_size, random_seed=0, max_depth=max_depth)
    else:
        return rfc.runRandomForest(file_loc, False, forest_size=forest_size, random_seed=0, max_depth=max_depth)

def randomForestBreastCancer(forest_sizes):
    # Source file location
    file_orig = "data/Wisconsin_breast_prognostic.csv"
    file_loc = file_orig

    times_multiprocessing = []
    times_single = []

    for forest_size in forest_sizes:
        print(f"\n\nRandom Forest Classifier on Wisconsin Breast Prognostic dataset with Forest Size: {forest_size}\n")

        # Measure the execution time without multiprocessing
        start_time_single = time.time()
        rfObjBreastCancer_single = runRandomForest(file_loc, False, forest_size=forest_size)
        randomForest_single, accuracy_single = rfObjBreastCancer_single.run()
        end_time_single = time.time()
        exec_time_single = end_time_single - start_time_single
        times_single.append(exec_time_single)
        print(f"Execution Time (Single): {exec_time_single:.2f} seconds\n")

        # Measure the execution time with multiprocessing
        start_time_multi = time.time()
        rfObjBreastCancer_multi = runRandomForest(file_loc, True, forest_size=forest_size)
        randomForest_multi, accuracy_multi = rfObjBreastCancer_multi.run()
        end_time_multi = time.time()
        exec_time_multi = end_time_multi - start_time_multi
        times_multiprocessing.append(exec_time_multi)
        print(f"Execution Time (Multiprocessing): {exec_time_multi:.2f} seconds\n")

        print(f"Forest Size: {forest_size}, Accuracy (Single): {accuracy_single}, Accuracy (Multiprocessing): {accuracy_multi}")


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(forest_sizes, times_single, marker='o', linestyle='-', color='b', label='Single Process')
    plt.plot(forest_sizes, times_multiprocessing, marker='s', linestyle='--', color='r', label='Multiprocessing')
    plt.xlabel('Forest Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs. Forest Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    forest_sizes = range(1, 1001, 100)
    randomForestBreastCancer(forest_sizes)
