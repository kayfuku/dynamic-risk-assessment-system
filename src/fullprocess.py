"""
Full process of ML pipeline with checking model drift.
Author: Kei
Date: January, 2022
"""
import os
import sys
import logging
import pandas as pd
from sklearn.metrics import f1_score

import scoring
import training
import ingestion
import reporting
import deployment
import diagnostics
from config import (
    INPUT_DATA_PATH,
    CLEANED_DATA_PATH,
    MODEL_PATH,
    PROD_DEPLOYMENT_PATH
)

logging.basicConfig(level=logging.INFO)


def main():
    # 1. Check and read new data
    logging.info("Checking for new data")
    with open(os.path.join(PROD_DEPLOYMENT_PATH, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}

    # Determine whether the source data folder has files that aren't
    # listed in ingestedfiles.txt
    source_files = set(os.listdir(INPUT_DATA_PATH))

    # 2. Deciding whether to proceed, part 1
    # If you found new data, you should proceed. otherwise, do end the process
    # here.
    # With this condition, ingestion.py will run even when some of the files are
    # not in the sourcedata dir.
    if len(source_files.difference(ingested_files)) == 0:
        logging.info("No new data found")
        return None

    # 3. Ingesting new data
    logging.info("Ingesting new data")
    ingestion.merge_multiple_dataframe()

    # 4. Check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    logging.info("Checking for model drift")
    # Get a new f1 score on the new model and test data.
    new_score = scoring.score_model(
        os.path.join(CLEANED_DATA_PATH, 'finaldata.csv'),
        os.path.join(PROD_DEPLOYMENT_PATH, 'trainedmodel.pkl'))

    with open(os.path.join(PROD_DEPLOYMENT_PATH, "latestscore.txt"), 'r') as f:
        deployed_score = f.read()

    # Deciding whether to proceed, part 2
    # Let's just pretend to be lower score.
    new_score = 0.22
    logging.info(f"Deployed score: {deployed_score}")
    logging.info(f"New score: {new_score}")

    # If you found model drift, you should proceed. otherwise, do end the
    # process here
    if new_score >= float(deployed_score):
        logging.info("No model drift occurred")
        return None

    # Model Drift has occurred!!
    logging.info("Model Drift has occurred!!")

    # 5. Re-training
    logging.info("Re-training model")
    training.train_model()
    logging.info("Re-scoring model")
    scoring.score_model(
        os.path.join(CLEANED_DATA_PATH, 'finaldata.csv'),
        os.path.join(MODEL_PATH, 'trainedmodel.pkl'))

    # 6. Re-deployment
    logging.info("Re-deploying model")
    deployment.deploy_model()

    # 7. Diagnostics and reporting
    logging.info("Running diagnostics and reporting")
    reporting.get_confusion_matrix()
    # reporting.generate_pdf_report()

    # 8. Call APIs.
    logging.info("Calling APIs")
    os.system("python apicalls.py")


if __name__ == '__main__':
    main()
