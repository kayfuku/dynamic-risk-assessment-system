"""
Deploy a model
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
import shutil

from config import CLEANED_DATA_PATH, MODEL_PATH, PROD_DEPLOYMENT_PATH

logging.basicConfig(level=logging.INFO)


def deploy_model():
    """
    Copy the latest model pickle file, the latestscore.txt, and
    the ingestedfiles.txt into the deployment directory
    """
    logging.info("Deploying trained model to production")
    logging.info(
        "Copying trainedmodel.pkl, ingestedfiles.txt and latestscore.txt")
    shutil.copy(
        os.path.join(CLEANED_DATA_PATH, 'ingestedfiles.txt'),
        PROD_DEPLOYMENT_PATH)
    shutil.copy(
        os.path.join(MODEL_PATH, 'trainedmodel.pkl'),
        PROD_DEPLOYMENT_PATH)
    shutil.copy(
        os.path.join(MODEL_PATH, 'latestscore.txt'),
        PROD_DEPLOYMENT_PATH)


# if __name__ == '__main__':
#     logging.info("Running deployment.py")
#     deploy_model()
