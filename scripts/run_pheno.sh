#!/usr/bin/env bash

# 在脚本中设置环境变量
#export DATA_ROOT="/home/ubuntu/hcy50662/output_mimic3"
export DATA_ROOT="/home/ubuntu/Virginia/output_mimic3"
#python -m src.train --config config/pheno_clinical_longformer.yaml
#python -m src.train --config config/pheno_timellm.yaml
#python -m src.train --config config/pheno_timellm_tsonly_freezebackbone.yaml
#python -m src.train --config config/pheno_gpt4mts.yaml
#python -m src.train --config config/pheno_clinical_bigbird.yaml
#python -m src.train --config config/pheno_belt.yaml
python -m src.train --config config/pheno_lstm.yaml









