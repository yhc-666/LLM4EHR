from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizer


def collate_fn(tokenizer: PreTrainedTokenizer, max_length: int):
    """Create a collate function for DataLoader."""

    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        docs: List[str] = []
        labels = []
        for example in batch:
            notes = example["text_list"]
            notes = notes[:5]
            joined = "\n".join(
                [f"### NOTE {i+1} ###\n{note}" for i, note in enumerate(notes)]
            )
            docs.append(joined)
            labels.append(example["label"])
        enc = tokenizer(
            docs,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if isinstance(labels[0], (list, tuple, torch.Tensor)):
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }

    return _fn

