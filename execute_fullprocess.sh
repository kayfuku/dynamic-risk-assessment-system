#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate risk-assess && \
cd /home/workspace/dynamic-risk-assessment-system/src && \
python apicalls.py
