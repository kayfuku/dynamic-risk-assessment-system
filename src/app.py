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


import diagnostics


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
        json: model predictions
    """
    filepath = request.get_json()['filepath']

    preds, _ = diagnostics.model_predictions(filepath)
    return jsonify(preds.tolist())


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    Scoring endpoint that runs the script scoring.py and
    gets the score of the deployed model

    Returns:
        str: model f1 score
    """
    output = subprocess.run(['python', 'scoring.py'],
                            capture_output=True).stdout
    output = re.findall(r'f1 score = \d*\.?\d+', output.decode())[0]
    return output


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Summary statistics endpoint that calls dataframe summary
    function from diagnostics.py

    Returns:
        json: summary statistics
    """
    return jsonify(diagnostics.dataframe_summary())


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    """
    Diagnostics endpoint thats calls missing_percentage, execution_time,
    and outdated_package_list from diagnostics.py

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
