#! /bin/bash
conda activate risk-assess && \
cd /home/workspace/dynamic-risk-assessment-system/src && \
python fullprocess.py && \
python apicalls.py
