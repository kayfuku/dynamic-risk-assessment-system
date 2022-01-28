"""
Diagnose model and data
Author: Kei
Date: January, 2022
"""
from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import pickle

from config import TEST_DATA_PATH, PROD_DEPLOYMENT_PATH, DATA_PATH

logging.basicConfig(level=logging.INFO)


def model_predictions():
    """
    Load test data and deployed model to predict on the test data.

    Return:
        y_pred: predictions
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


def dataframe_summary():
    """
    Load finaldata.csv and calculates mean, median and std on numerical column.

    Returns:
        statistics: dict, key: column, value: dict of statistics for that column.
    """
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes(include='number')

    logging.info("Calculating statistics for data")
    statistics = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()

        statistics[col] = {'mean': mean, 'median': median, 'std': std}

    return statistics


def missing_percentage():
    """
    Calculate percentage of missing data for each column in finaldata.csv

    Returns:
        list[dict]: Each dict contains column name and percentage
    """
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))

    logging.info("Calculating missing data percentage")
    missing_data_percentage = {}
    for col in data_df.columns:
        percentage = (data_df[col].isna().sum() / data_df.shape[0]) * 100
        missing_data_percentage[col] = {'percentage': percentage}

    return missing_data_percentage


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

    statistics = dataframe_summary()
    print(statistics)

    missing_data_percentage = missing_percentage()
    print(missing_data_percentage)

    execution_time()
    outdated_packages_list()
