"""
Diagnose model and data
Author: Kei
Date: January, 2022
"""
import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import pickle

from config import TEST_DATA_PATH, PROD_DEPLOYMENT_PATH

logging.basicConfig(level=logging.INFO)


def model_predictions():
    """
    Load test data and deployed model to predict on the test data.
    """
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))
    X_df = test_df.drop(['corporation', 'exited'], axis=1)

    logging.info("Loading deployed model")
    with open(os.path.join(PROD_DEPLOYMENT_PATH, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    logging.info("Making predictions on data")
    y_pred = model.predict(X_df)

    return y_pred

# Function to get summary statistics


def dataframe_summary():
    # calculate summary statistics here
    return  # return value should be a list containing all summary statistics

# Function to get timings


def execution_time():
    # calculate timing of training.py and ingestion.py
    return  # return a list of 2 timing values in seconds

# Function to check dependencies


def outdated_packages_list():
    # get a list of
    pass


if __name__ == '__main__':
    y_pred = model_predictions()
    print(y_pred)

    dataframe_summary()
    execution_time()
    outdated_packages_list()
