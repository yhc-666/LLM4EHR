from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


class InconsistentSplittingParamsException(Exception):
    pass


def split_overlapping(tensor: Tensor, chunk_size: int, stride: int, minimal_chunk_length: int) -> List[Tensor]:
    if chunk_size > 510:
        raise InconsistentSplittingParamsException("Size of each chunk cannot be bigger than 510!")
    if minimal_chunk_length > chunk_size:
        raise InconsistentSplittingParamsException("Minimal length cannot be bigger than size!")
    if stride > chunk_size:
        raise InconsistentSplittingParamsException(
            "Stride cannot be bigger than size! Chunks must overlap or be near each other!"
        )
    result = [tensor[i : i + chunk_size] for i in range(0, len(tensor), stride)]
    if len(result) > 1:
        result = [x for x in result if len(x) >= minimal_chunk_length]
    return result


def stack_tokens_from_all_chunks(input_id_chunks: List[Tensor], mask_chunks: List[Tensor]) -> tuple[Tensor, Tensor]:
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)
    return input_ids.long(), attention_mask.int()


def add_padding_tokens(input_id_chunks: List[Tensor], mask_chunks: List[Tensor]) -> None:
    for i in range(len(input_id_chunks)):
        pad_len = 512 - input_id_chunks[i].shape[0]
        if pad_len > 0:
            input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([0] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0] * pad_len)])


def add_special_tokens_at_beginning_and_end(input_id_chunks: List[Tensor], mask_chunks: List[Tensor]) -> None:
    if len(input_id_chunks) == 0:
        input_id_chunks.append(torch.Tensor([101, 102]))
        mask_chunks.append(torch.Tensor([1, 1]))
        return
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([Tensor([101]), input_id_chunks[i], Tensor([102])])
        mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])


def split_tokens_into_smaller_chunks(
    tokens,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
) -> tuple[List[Tensor], List[Tensor]]:
    input_id_chunks = split_overlapping(tokens["input_ids"][0], chunk_size, stride, minimal_chunk_length)
    mask_chunks = split_overlapping(tokens["attention_mask"][0], chunk_size, stride, minimal_chunk_length)
    return input_id_chunks, mask_chunks


def tokenize_text_with_truncation(text: str, tokenizer: PreTrainedTokenizerBase, maximal_text_length: int):
    tokens = tokenizer(
        text, add_special_tokens=False, max_length=maximal_text_length, truncation=True, return_tensors="pt"
    )
    return tokens


def tokenize_whole_text(text: str, tokenizer: PreTrainedTokenizerBase):
    tokens = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors="pt")
    return tokens


def transform_single_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    maximal_text_length: Optional[int],
) -> tuple[Tensor, Tensor]:
    if maximal_text_length:
        tokens = tokenize_text_with_truncation(text, tokenizer, maximal_text_length)
    else:
        tokens = tokenize_whole_text(text, tokenizer)
    input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tokens, chunk_size, stride, minimal_chunk_length)
    add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
    add_padding_tokens(input_id_chunks, mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
    return input_ids, attention_mask


def transform_list_of_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    maximal_text_length: Optional[int] = None,
):
    model_inputs = [
        transform_single_text(text, tokenizer, chunk_size, stride, minimal_chunk_length, maximal_text_length)
        for text in texts
    ]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class BeltOutputs:
    logits: Tensor
    loss: Optional[Tensor]


class BeltForLongTexts(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        chunk_size: int = 128,
        stride: int = 64,
        minimal_chunk_length: int = 50,
        pooling_strategy: str = "mean",
        maximal_text_length: Optional[int] = None,
        use_4bit: bool = False,
        lora_cfg: Optional[Dict[str, int]] = None,
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

        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            torch_dtype=torch.float32,
            device_map="auto",
            use_cache=False,
        )

        if lora_cfg is not None:
            lora_config = LoraConfig(**lora_cfg)
            self.model = get_peft_model(self.model, lora_config)

        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.cls_token

        hidden = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

        if pooling_strategy not in {"mean", "max"}:
            raise ValueError("Unknown pooling strategy!")

        self.chunk_size = chunk_size
        self.stride = stride
        self.minimal_chunk_length = minimal_chunk_length
        self.pooling_strategy = pooling_strategy
        self.maximal_text_length = maximal_text_length

    def forward(
        self,
        input_ids: List[Tensor],
        attention_mask: List[Tensor],
        labels: Optional[Tensor] = None,
    ) -> BeltOutputs:
        number_of_chunks = [len(x) for x in input_ids]
        combined_ids = torch.cat([t.to(self.classifier.weight.device) for t in input_ids], dim=0)
        combined_mask = torch.cat([t.to(self.classifier.weight.device) for t in attention_mask], dim=0)

        outputs = self.model(input_ids=combined_ids, attention_mask=combined_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        logits_split = logits.split(number_of_chunks, dim=0)
        if self.pooling_strategy == "mean":
            pooled_logits = torch.stack([torch.mean(x, dim=0) for x in logits_split])
        else:
            pooled_logits = torch.stack([torch.max(x, dim=0)[0] for x in logits_split])

        loss = None
        if labels is not None:
            if pooled_logits.size(1) == 1 or labels.ndim == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(pooled_logits.squeeze(), labels.float())
            else:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(pooled_logits, labels.float())

        return BeltOutputs(logits=pooled_logits, loss=loss)
