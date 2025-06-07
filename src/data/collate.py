from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizer


def collate_fn(tokenizer: PreTrainedTokenizer, max_length: int, model_type: str = "llama"):
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
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }
        if model_type == "clinicallongformer":
            global_attention_mask = torch.zeros_like(attention_mask)           # (b, L)
            seq_len = global_attention_mask.size(1)
            idx = torch.arange(0, seq_len, 128, device=global_attention_mask.device)
            # 仅在有效 token 上置 1
            valid_idx = (attention_mask[:, idx] == 1).long()    
            global_attention_mask[:, idx] = valid_idx

            # 保底：始终给 CLS 位置全局注意力
            global_attention_mask[:, 0] = 1
            batch_dict["global_attention_mask"] = global_attention_mask
        return batch_dict

    return _fn

