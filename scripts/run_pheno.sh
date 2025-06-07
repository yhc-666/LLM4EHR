#!/usr/bin/env bash

# 在脚本中设置环境变量
export DATA_ROOT="/home/ubuntu/hcy50662/output_mimic3"
python -m src.train --config config/pheno_clinical_longformer.yaml








