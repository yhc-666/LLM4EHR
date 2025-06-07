from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


@dataclass
class LongformerOutputs:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class ClinicalLongformerPool(nn.Module):
    """Clinical Longformer with configurable pooling head."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        use_4bit: bool = False,
        lora_cfg: Optional[Dict[str, int]] = None,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        quant_cfg = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if use_4bit
            else None
        )

        self.pooling = pooling.lower()
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")


        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,
        )

        if lora_cfg is not None:
            lora_config = LoraConfig(**lora_cfg)
            self.model = get_peft_model(self.model, lora_config)

        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.cls_token

        hidden = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        global_interval: int = 128,
    ) -> LongformerOutputs:
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled = (last_hidden * mask).sum(1) / mask.sum(1)
        else:
            pooled = last_hidden[:, 0]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if logits.size(1) == 1 or labels.ndim == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels.float())
        return LongformerOutputs(logits=logits, loss=loss)
