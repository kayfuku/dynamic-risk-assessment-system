"""
Flask app APIs
Author: Kei
Date: January, 2022
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess

from config import TEST_DATA_PATH, PROD_DEPLOYMENT_PATH
import diagnostics
import scoring


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


@app.route('/')
def index():
    return "Hello World!"


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    Load data given the file path and call the prediction function in diagnostics.py

    Returns:
        json, predictions
    """
    data_path = request.get_json()['filepath']
    _, preds = diagnostics.model_predictions(data_path)

    return jsonify(preds.tolist())


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    Run the script scoring.py and
    gets the score of the deployed model

    Returns:
        str, model f1 score
    """
    output = scoring.score_model(
        os.path.join(TEST_DATA_PATH, 'testdata.csv'),
        os.path.join(PROD_DEPLOYMENT_PATH, 'trainedmodel.pkl'))

    return str(output)


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Call dataframe summary function from diagnostics.py

    Returns:
        json: summary statistics
    """
    return jsonify(diagnostics.dataframe_summary())


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    """
    Call missing_percentage, execution_time, and outdated_package_list
    from diagnostics.py

    Returns:
        dict: missing percentage, execution time and outdated packages
    """
    missing = diagnostics.missing_percentage()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()

    ret = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }

    return jsonify(ret)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
