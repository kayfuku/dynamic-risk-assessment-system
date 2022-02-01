"""
Ingest data
Author: Kei
Date: January, 2022
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

from config import INPUT_DATA_PATH, CLEANED_DATA_PATH

logging.basicConfig(level=logging.INFO)


def merge_multiple_dataframe():
    """
    Function for data ingestion. Merge multiple dataframe in the INPUT_DATA_PATH,
    INPUT_DATA_PATH, and dedup the duplicates and write it to CLEANED_DATA_PATH.
    Also, save metadata in the CLEANED_DATA_PATH.
    """

    df = pd.DataFrame()
    file_names = []

    logging.info(f"Reading files from {INPUT_DATA_PATH}")
    for file in os.listdir(INPUT_DATA_PATH):
        file_path = os.path.join(INPUT_DATA_PATH, file)
        df_tmp = pd.read_csv(file_path)
        df = df.append(df_tmp, ignore_index=True)

        file_names.append(file)

    logging.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    logging.info("Saving ingested metadata")
    # Keep track of file names that was used to create the data.
    with open(os.path.join(CLEANED_DATA_PATH, 'ingestedfiles.txt'), "w") as file:
        file.write(
            f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(file_names))

    logging.info("Saving ingested data")
    df.to_csv(os.path.join(CLEANED_DATA_PATH, 'finaldata.csv'), index=False)


# if __name__ == '__main__':
#     logging.info("Running ingestion.py")
#     merge_multiple_dataframe()
