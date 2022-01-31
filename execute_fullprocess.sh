#! /bin/bash
conda activate risk_assess && \
cd /home/workspace/dynamic-risk-assessment-system/src && \
python fullprocess.py && \
python apicalls.py
