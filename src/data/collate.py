from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizer
from ..models.belt import transform_list_of_texts

from ..utils import BaseConfig
import numpy as np


def collate_fn(
    tokenizer: PreTrainedTokenizer | None,
    cfg: BaseConfig,
):
    """Create a collate function for DataLoader.

    Parameters
    ----------
    tokenizer : :class:`~transformers.PreTrainedTokenizer`
        Tokenizer used to encode text.
    cfg : :class:`~src.utils.BaseConfig`
        Experiment configuration. Must contain ``max_seq_len`` and ``model_type`` fields.
    """

    max_length = cfg.max_seq_len
    model_type = cfg.model_type.lower()
    use_text = True
    if model_type == "timellm":
        use_text = getattr(cfg, "enable_text", True)

    # Some smaller test tokenizers may not define a pad token. ``padding``
    # will fail in that case, so fall back to using the EOS token.
    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if model_type in {"lstm", "cnn"}:
            """Collate regularised time-series and labels for LSTM/CNN."""
            labels, ts = [], []
            for ex in batch:
                labels.append(ex["label"])
                ts.append(torch.tensor(ex["reg_ts"], dtype=torch.float32))
            labels_array = np.array(labels, dtype=np.float32)
            labels_tensor = torch.from_numpy(labels_array)
            return {
                "reg_ts": torch.stack(ts),  # (B, T, F)
                "labels": labels_tensor,  # (B,) or (B, num_labels)
            }
        if model_type == "gpt4mts":
            labels, ts, tokens = [], [], []
            use_text_prefix = getattr(cfg, "enable_text", True)
            for ex in batch:
                labels.append(ex["label"])
                ts.append(torch.tensor(ex["reg_ts"], dtype=torch.float32))
                if use_text_prefix:
                    notes = ex["text_list"][:5]
                    if not notes:
                        notes = [" "]
                    enc = tokenizer(
                        notes,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    tokens.append(enc)
            labels_array = np.array(labels, dtype=np.float32)
            labels_tensor = torch.from_numpy(labels_array)
            batch_dict = {
                "reg_ts": torch.stack(ts),
                "labels": labels_tensor,
            }
            if use_text_prefix:
                batch_dict["summary_tokens"] = tokens
                # tokens: List of dicts, length = batch_size
                # each dict contains "input_ids"(num_notes, max_length), "attention_mask"(num_notes, max_length)
            else:
                batch_dict["summary_tokens"] = None
            return batch_dict
        if model_type == "belt":
            docs, labels = [], []
            for ex in batch:
                notes = ex["text_list"][-5:]
                joined = "\n".join(
                    [f"### NOTE {i+1} ###\n{note}" for i, note in enumerate(notes)]
                )
                docs.append(joined)
                labels.append(ex["label"])
            tokens = transform_list_of_texts(
                docs,
                tokenizer,
                cfg.chunk_size,
                cfg.stride,
                cfg.minimal_chunk_length,
                getattr(cfg, "maximal_text_length", None),
            )
            labels_array = np.array(labels, dtype=np.float32)
            labels_tensor = torch.from_numpy(labels_array)
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": labels_tensor,
            }

        docs: List[str] = []
        labels = []
        for example in batch:
            notes = example["text_list"] if use_text else []
            notes = notes[:5]
            if notes:
                joined = "\n".join(
                    [f"### NOTE {i+1} ###\n{note}" for i, note in enumerate(notes)]
                )
            else:
                joined = ""
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
            # 优化：先转换为numpy数组再创建tensor
            labels_array = np.array(labels, dtype=np.float32)
            labels_tensor = torch.from_numpy(labels_array)
        else:
            # 优化：先转换为numpy数组再创建tensor
            labels_array = np.array(labels, dtype=np.float32)
            labels_tensor = torch.from_numpy(labels_array)
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }
        if model_type == "timellm":
            ts = [torch.tensor(ex["reg_ts"], dtype=torch.float32) for ex in batch]
            batch_dict["reg_ts"] = torch.stack(ts)
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
        if model_type == "clinicalbigbird":
            pass  # identical to Llama behavior
        return batch_dict

    return _fn

