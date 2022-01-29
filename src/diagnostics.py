"""
Diagnose model and data
Author: Kei
Date: January, 2022
"""
from textwrap import indent
from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import pickle
import subprocess

from config import TEST_DATA_PATH, PROD_DEPLOYMENT_PATH, DATA_PATH

logging.basicConfig(level=logging.INFO)


def model_predictions(filepath):
    """
    Load test data and deployed model to predict on the test data.

    Return:
        y_pred: predictions
    """
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))
    y_df = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    logging.info("Loading deployed model")
    with open(os.path.join(PROD_DEPLOYMENT_PATH, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    logging.info("Making predictions on data")
    y_pred = model.predict(X_df)

    return y_pred, y_df


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


def _ingestion_timing():
    """
    Run ingestion.py script and measures execution time

    Returns:
        timing: float, running time
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing


def _training_timing():
    """
    Run training.py script and measures execution time

    Returns:
        timing: float, running time
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing


def execution_time():
    """
    Get average execution time for data ingestion and model training
    by running 10 times for each.

    Returns:
        running_time_means: means of execution times for each script
    """
    logging.info("Calculating time for ingestion.py")
    ingestion_time = []
    for _ in range(5):
        time = _ingestion_timing()
        ingestion_time.append(time)

    logging.info("Calculating time for training.py")
    training_time = []
    for _ in range(5):
        time = _training_timing()
        training_time.append(time)

    running_time_means = [
        {'ingest_time_mean': np.mean(ingestion_time)},
        {'train_time_mean': np.mean(training_time)}
    ]

    return running_time_means


def outdated_packages_list():
    """
    Check dependencies from requirements.txt using pip-outdated
    which checks each package if it is outdated or not.

    Returns:
        dep: stdout of the pip-outdated command
    """
    logging.info("Checking outdated dependencies")
    dependencies = subprocess.run([
        'pip-outdated', '../requirements.txt'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8')

    dep = dependencies.stdout
    dep = dep.translate(str.maketrans('', '', ' \t\r'))
    dep = dep.split('\n')
    dep = [dep[3]] + dep[5:-3]
    dep = [s.split('|')[1:-1] for s in dep]

    return dep

    # dependencies = subprocess.check_output(
    #     ['pip', 'list', '--outdated'],
    # )
    # return dependencies


if __name__ == '__main__':
    y_pred = model_predictions(os.path.join(TEST_DATA_PATH, 'testdata.csv'))
    print(y_pred)

    statistics = dataframe_summary()
    print(json.dumps(statistics, indent=4))

    missing_data_percentage = missing_percentage()
    print(json.dumps(missing_data_percentage, indent=4))

    running_time_means = execution_time()
    print(running_time_means)

    dependencies = outdated_packages_list()
    for row in dependencies:
        print(row)
