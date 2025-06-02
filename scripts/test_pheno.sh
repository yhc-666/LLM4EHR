#!/usr/bin/env bash

# pheno任务测试脚本
# 用于评估训练好的pheno模型性能


export DATA_ROOT="/home/ubuntu/hcy50662/output_mimic3/pheno"
# 运行测试
python -m src.test --config config/pheno.yaml
