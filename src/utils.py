from __future__ import annotations

import os
import random
import yaml
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, Type

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


def to_device(obj: Any, device: torch.device):
    """Recursively move tensors inside ``obj`` to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


@dataclass
class BaseConfig:
    """Dataclass wrapper for common experiment settings."""

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


@dataclass
class LlamaConfig(BaseConfig):
    """Configuration for Llama models."""

    freeze: bool = False

@dataclass
class ClinicalLongformerConfig(BaseConfig):
    """Configuration for Clinical-Longformer."""

    pooling: str = "mean"


@dataclass
class ClinicalBigBirdConfig(BaseConfig):
    """Configuration for Clinical-BigBird."""

    pooling: str = "mean"


@dataclass
class BeltConfig(BaseConfig):
    """Configuration for BELT models."""
    chunk_size: int = 128
    stride: int = 64
    minimal_chunk_length: int = 50
    pooling_strategy: str = "mean"
    maximal_text_length: int | None = None

@dataclass
class LSTMConfig(BaseConfig):
    """Configuration for the LSTM baseline."""

    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1


@dataclass
class CNNConfig(BaseConfig):
    """Configuration for the CNN baseline."""

    num_filters: int = 64
    kernel_size: int = 3
    dropout: float = 0.1

@dataclass
class TimeLLMConfig(BaseConfig):
    """Configuration for TimeLLM models."""

    d_model: int | None = None
    patch_len: int = 8
    stride: int = 8
    n_heads: int = 8
    freezebasemodel: bool = False
    enable_text: bool = True


@dataclass
class GPT4MTSConfig(BaseConfig):
    """Configuration for GPT4MTS models."""
    seq_len: int = 24
    patch_size: int = 8
    stride: int = 4
    gpt_layers: int = 6
    d_model: int = 768
    freeze: bool = False
    pretrain: bool = True
    revin: bool = False
    classifier_head: str = "linear"
    enable_text: bool = True


def parse_config_yaml(path: str) -> BaseConfig:
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
    model_type = data.get("model_type", "llama").lower()
    cfg_map: Dict[str, Type[BaseConfig]] = {
        "llama": LlamaConfig,
        "clinicallongformer": ClinicalLongformerConfig,
        "clinicalbigbird": ClinicalBigBirdConfig,
        "timellm": TimeLLMConfig,
        "gpt4mts": GPT4MTSConfig,
        "belt": BeltConfig,
        "lstm": LSTMConfig,
        "cnn": CNNConfig,
    }
    cfg_cls = cfg_map.get(model_type, BaseConfig)
    allowed = {f.name for f in fields(cfg_cls)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return cfg_cls(**filtered)

