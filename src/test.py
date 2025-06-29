from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .data.loader import MIMICDataset
from .data.collate import collate_fn
from .models.llama_mean import LlamaMeanPool
from .models.clinicallongformer import ClinicalLongformerPool
from .models.clinicalbigbird import ClinicalBigBirdPool
from .models.timellm import TimeLLM
from .models.gpt4mts import GPT4MTS
from .models.belt import BeltForLongTexts
from .models.lstm import LSTMClassifier
from .models.cnn import CNNClassifier
from .metrics import binary_metrics, multilabel_metrics
from .utils import parse_config_yaml, set_seed, to_device


def main(config_path: str) -> None:
    cfg = parse_config_yaml(config_path)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    set_seed(42)

    if cfg.model_type == "llama":
        model = LlamaMeanPool(
            cfg.pretrained_meta_model,
            cfg.num_labels,
            use_4bit=cfg.use_4bit,
            lora_cfg=cfg.lora,
            freeze=cfg.freeze,
        )
    elif cfg.model_type == "clinicallongformer":
        model = ClinicalLongformerPool(
            cfg.pretrained_meta_model,
            cfg.num_labels,
            use_4bit=cfg.use_4bit,
            lora_cfg=cfg.lora,
            pooling=cfg.pooling,
        )
    elif cfg.model_type == "clinicalbigbird":
        model = ClinicalBigBirdPool(
            cfg.pretrained_meta_model,
            cfg.num_labels,
            use_4bit=cfg.use_4bit,
            lora_cfg=cfg.lora,
            pooling=cfg.pooling,
        )
    elif cfg.model_type == "timellm":
        model = TimeLLM(
            cfg.pretrained_meta_model,
            cfg.num_labels,
            use_4bit=cfg.use_4bit,
            lora_cfg=cfg.lora,
            d_model=cfg.d_model,
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            n_heads=cfg.n_heads,
            freeze_base_model=cfg.freezebasemodel,
            enable_text=cfg.enable_text,
        )
    elif cfg.model_type == "gpt4mts":
        model = GPT4MTS(
            cfg.pretrained_meta_model,
            cfg.num_labels,
            seq_len=cfg.seq_len,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            gpt_layers=cfg.gpt_layers,
            d_model=cfg.d_model,
            freeze=cfg.freeze,
            pretrain=cfg.pretrain,
            revin=cfg.revin,
            classifier_head=cfg.classifier_head,
            enable_text_as_prefix=cfg.enable_text_as_prefix,
        )
    elif cfg.model_type == "lstm":
        model = LSTMClassifier(
            input_dim=34,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_labels=cfg.num_labels,
            dropout=cfg.dropout,
        )
    elif cfg.model_type == "cnn":
        model = CNNClassifier(
            input_dim=34,
            num_filters=cfg.num_filters,
            kernel_size=cfg.kernel_size,
            num_labels=cfg.num_labels,
            dropout=cfg.dropout,
        )
    elif cfg.model_type == "belt":
        model = BeltForLongTexts(
            cfg.pretrained_meta_model,
            cfg.num_labels,
            chunk_size=cfg.chunk_size,
            stride=cfg.stride,
            minimal_chunk_length=cfg.minimal_chunk_length,
            pooling_strategy=cfg.pooling_strategy,
            maximal_text_length=cfg.maximal_text_length,
            use_4bit=cfg.use_4bit,
            lora_cfg=cfg.lora,
        )
    else:
        raise ValueError("unknown model_type")
    
    path = Path(cfg.save_path) / "best.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing, f"有 {len(missing)} 个真实参数缺失！"
        print(f"忽略 {len(unexpected)} 个 bitsandbytes 缓冲区")
    tokenizer = model.tokenizer

    test_ds = MIMICDataset(cfg, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, cfg),
    )

    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: to_device(v, accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = accelerator.gather(outputs.logits).cpu().float().numpy()
            label = accelerator.gather(batch["labels"]).cpu().float().numpy()
            preds.append(logits)
            labels.append(label)
    # 正确地连接不同大小的批次
    pred = np.concatenate(preds, axis=0)
    lab = np.concatenate(labels, axis=0)
    if cfg.task == "ihm":
        metrics = binary_metrics(pred, lab)
    else:
        metrics = multilabel_metrics(pred, lab)
    accelerator.print({k: f"{v:.4f}" for k, v in metrics.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Llama for MIMIC tasks")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)

