"""
Model Reporting
Author: Kei
Date: January, 2022
"""
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns

import json
import os
import logging


import diagnostics
from config import TEST_DATA_PATH, MODEL_PATH


logging.basicConfig(level=logging.INFO)


def get_confusion_matrix():
    """
    Plot a confusion matrix using the test data and the deployed model
    """
    logging.info("Predicting test data")
    filepath = os.path.join(TEST_DATA_PATH, 'testdata.csv')
    y_pred, y_true = diagnostics.model_predictions(filepath)

    logging.info("Plotting and saving confusion matrix")
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()
    sub_cm.set_title("Model Confusion Matrix")
    fig_cm.savefig(os.path.join(MODEL_PATH, 'confusionmatrix.png'))


if __name__ == '__main__':
    get_confusion_matrix()
