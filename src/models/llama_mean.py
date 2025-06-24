from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


@dataclass
class LlamaOutputs:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class LlamaMeanPool(nn.Module):
    """Llama model with masked mean pooling head.

    Parameters
    ----------
    model_name:
        Name of the pretrained Llama model.
    num_labels:
        Number of prediction labels.
    use_4bit:
        Whether to load the base model in 4-bit mode.
    lora_cfg:
        Optional LoRA configuration.
    freeze:
        If ``True``, freeze all Llama parameters and only train the
        classification head.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        use_4bit: bool = False,
        lora_cfg: Optional[Dict[str, int]] = None,
        freeze: bool = False,
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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,
        )

        if lora_cfg is not None:
            lora_config = LoraConfig(**lora_cfg)
            self.model = get_peft_model(self.model, lora_config)

        self.model.gradient_checkpointing_enable()  # 节省大量激活显存!!!(影响最大)  
        self.model.enable_input_require_grads()     # 让 checkpoint 反向链路完整

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        
        # Llama 模型的 tokenizer 没有默认的 padding token，需要手动设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # left padding 更合适
        self.tokenizer.padding_side = "left"

        hidden = self.model.config.hidden_size

        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> LlamaOutputs:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1)
        pooled = (last_hidden * mask).sum(1) / mask.sum(1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if logits.size(1) == 1 or labels.ndim == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels.float())
        return LlamaOutputs(logits=logits, loss=loss)

