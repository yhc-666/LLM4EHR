from __future__ import annotations

import os
import random
import yaml
import re
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model state_dict to ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


@dataclass
class Config:
    """Dataclass wrapper for experiment configuration."""

    train_pkl: str
    val_pkl: str
    test_pkl: str
    save_path: str
    max_seq_len: int
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    grad_accum: int
    pretrained_meta_model: str
    use_4bit: bool
    lora: Dict[str, Any] | None
    model_type: str
    task: str
    num_labels: int
    wandb: bool
    mixed_precision: str = "no"  # 支持 "no", "fp16", "bf16"
    pooling: str = "mean"


def parse_config_yaml(path: str) -> Config:
    """Parse YAML config file and expand environment variables."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 只对包含 ${...} 模式的字符串进行环境变量展开
    for key, value in data.items():
        if isinstance(value, str) and "${" in value:
            data[key] = os.path.expandvars(value)
        elif isinstance(value, str) and re.match(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$', value):
            try:
                data[key] = float(value)
            except ValueError:
                pass  # 如果转换失败，保持原值
    return Config(**data)

