"""
Score a model
Author: Kei
Date: January, 2022
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging
from config import MODEL_PATH, TEST_DATA_PATH

logging.basicConfig(level=logging.INFO)


def score_model(data_path, model_path):
    """
    Loads a trained model and the test data, and calculate an F1 score
    for the model on the test data and saves the result to
    the latestscore.txt file
    """
    logging.info(f"Loading data from {data_path}")
    test_df = pd.read_csv(data_path)

    logging.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logging.info("Preparing data")
    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    logging.info("Predicting")
    y_pred = model.predict(X_df)
    f1 = f1_score(y_true, y_pred)
    print(f"f1 score = {f1}")

    logging.info("Saving scores to text file")
    with open(os.path.join(MODEL_PATH, 'latestscore.txt'), 'w') as f:
        f.write(str(f1))

    return f1


# if __name__ == '__main__':
#     logging.info("Running scoring.py")
#     score_model(os.path.join(TEST_DATA_PATH, 'testdata.csv'),
#                 os.path.join(MODEL_PATH, 'trainedmodel.pkl'))
