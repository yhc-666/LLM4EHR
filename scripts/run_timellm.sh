#!/usr/bin/env bash

export DATA_ROOT="/home/ubuntu/hcy50662/output_mimic3"
python -m src.train --config config/pheno_timellm.yaml
