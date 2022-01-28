from distutils.command.config import config
import os
import json

with open('../config.json', 'r') as file:
    config = json.load(file)

INPUT_FOLDER_PATH = os.path.join(
    os.path.abspath('../'),
    'data',
    config['input_folder_path'])
DATA_PATH = os.path.join(
    os.path.abspath('../'),
    'data',
    config['output_folder_path'])
TEST_DATA_PATH = os.path.join(
    os.path.abspath('../'),
    'data',
    config['test_data_path'])
MODEL_PATH = os.path.join(
    os.path.abspath('../'),
    'model',
    config['output_model_path'])
PROD_DEPLOYMENT_PATH = os.path.join(os.path.abspath(
    '../'), 'model', config['prod_deployment_path'])
