#! /bin/bash
conda activate risk_access && \
cd /home/workspace/dynamic-risk-assessment-system/src && \
python fullprocess.py && \
python apicalls.py
