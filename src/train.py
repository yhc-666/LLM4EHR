from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data.loader import MIMICDataset
from .data.collate import collate_fn
from .models.llama_mean import LlamaMeanPool
from .metrics import binary_metrics, multilabel_metrics
from .utils import Config, parse_config_yaml, save_checkpoint, set_seed


def evaluate(
    accelerator: Accelerator, model: LlamaMeanPool, dataloader: DataLoader, cfg: Config
) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = accelerator.gather(outputs.logits).cpu().numpy()
            labels = accelerator.gather(batch["labels"]).cpu().numpy()
            preds.append(logits)
            targets.append(labels)
    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    if cfg.task == "ihm":
        metrics = binary_metrics(pred_arr, target_arr)
        f1 = metrics["F1"]
    else:
        metrics = multilabel_metrics(pred_arr, target_arr)
        f1 = metrics["macro_F1"]
    accelerator.print({k: f"{v:.4f}" for k, v in metrics.items()})
    return f1


def main(config_path: str) -> None:
    cfg = parse_config_yaml(config_path)
    accelerator = Accelerator()
    set_seed(42)

    model = LlamaMeanPool(
        cfg.pretrained_meta_model,
        cfg.num_labels,
        use_4bit=cfg.use_4bit,
        lora_cfg=cfg.lora,
    )
    tokenizer = model.tokenizer

    train_ds = MIMICDataset(cfg.train_pkl, cfg.task)
    val_ds = MIMICDataset(cfg.val_pkl, cfg.task)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, cfg.max_seq_len),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, cfg.max_seq_len),
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    num_training_steps = len(train_loader) * cfg.num_epochs
    num_warmup = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_training_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_f1 = 0.0
    for epoch in range(cfg.num_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if (step + 1) % cfg.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
        f1 = evaluate(accelerator, model, val_loader, cfg)
        if accelerator.is_main_process and f1 > best_f1:
            best_f1 = f1
            path = Path(cfg.save_path) / "best.pt"
            save_checkpoint(accelerator.unwrap_model(model), str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama for MIMIC tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()
    main(args.config)

