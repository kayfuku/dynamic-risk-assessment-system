"""
Train a model
Author: Kei
Date: January, 2022
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

from config import CLEANED_DATA_PATH, MODEL_PATH

logging.basicConfig(level=logging.INFO)


def train_model():
    """
     Train a model on ingested data and
     and saves the model
     """
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(CLEANED_DATA_PATH, 'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    logging.info("Training model")
    model.fit(X_df, y_df)

    logging.info("Saving trained model")
    with open(os.path.join(MODEL_PATH, 'trainedmodel.pkl'), 'wb') as f:
        pickle.dump(model, f)


# if __name__ == '__main__':
#     logging.info("Running training.py")
#     train_model()
