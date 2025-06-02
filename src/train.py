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
from tqdm.auto import tqdm
import wandb

from .data.loader import MIMICDataset
from .data.collate import collate_fn
from .models.llama_mean import LlamaMeanPool
from .metrics import binary_metrics, multilabel_metrics
from .utils import Config, parse_config_yaml, save_checkpoint, set_seed


def evaluate(
    accelerator: Accelerator, model: LlamaMeanPool, dataloader: DataLoader, cfg: Config
) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []
    loader = tqdm(
        dataloader,
        disable=not accelerator.is_local_main_process,
        desc="Validation",
    )
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = accelerator.gather(outputs.logits).cpu().float().numpy()
            labels = accelerator.gather(batch["labels"]).cpu().float().numpy()
            preds.append(logits)
            targets.append(labels)
    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    if cfg.task == "ihm":
        metrics = binary_metrics(pred_arr, target_arr)
    else:
        metrics = multilabel_metrics(pred_arr, target_arr)
    accelerator.print({k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def main(config_path: str) -> None:
    cfg = parse_config_yaml(config_path)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    set_seed(42)

    if accelerator.is_main_process:
        if cfg.mixed_precision != "no":
            print(f"train with mixed precision: {cfg.mixed_precision}")
        else:
            print("train with full precision")

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
    global_step = 0
    if cfg.wandb and wandb is not None and accelerator.is_main_process:
        wandb.init(project="LLM4EHR", config=cfg.__dict__)

    for epoch in range(cfg.num_epochs):
        model.train()
        progress = tqdm(
            train_loader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
        )
        for step, batch in enumerate(progress):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if (step + 1) % cfg.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            progress.set_postfix(loss=loss.item())
            if cfg.wandb and wandb is not None and accelerator.is_main_process:
                wandb.log({"train_loss": loss.item(), "step": global_step})
            global_step += 1

        val_metrics = evaluate(accelerator, model, val_loader, cfg)
        if cfg.wandb and wandb is not None and accelerator.is_main_process:
            wandb.log({f"val_{k}": v for k, v in val_metrics.items()}, step=global_step)
        f1 = val_metrics["F1"] if cfg.task == "ihm" else val_metrics["macro_F1"]
        if accelerator.is_main_process and f1 > best_f1:
            best_f1 = f1
            path = Path(cfg.save_path) / "best.pt"
            save_checkpoint(accelerator.unwrap_model(model), str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama for MIMIC tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()
    main(args.config)

