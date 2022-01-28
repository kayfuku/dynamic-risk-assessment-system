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
