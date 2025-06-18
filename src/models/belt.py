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
    texts: List[str],  # 输入文本列表，长度为 batch_size
    tokenizer: PreTrainedTokenizerBase,  # 预训练的tokenizer
    chunk_size: int,  # 每个块的大小（token数量，不包括特殊token）
    stride: int,  # 滑动窗口的步长
    minimal_chunk_length: int,  # 最小块长度
    maximal_text_length: Optional[int] = None,  # 最大文本长度限制（可选）
):
    """
    将文本列表转换为模型输入格式，支持长文本分块处理
    
    输入Shape:
        texts: List[str] of length batch_size - 原始文本列表
        
    输出Shape:
        返回字典包含:
        - input_ids: List[Tensor] of length batch_size
          每个元素形状为 (num_chunks_i, 512)，其中 num_chunks_i 为第i个文本/note分成的的块数量
        - attention_mask: List[Tensor] of length batch_size  
          每个元素形状为 (num_chunks_i, 512)，对应input_ids的注意力掩码
          
    note: 每个文本的块数量 num_chunks_i 可能不同，取决于文本长度和分块参数
    """
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
    """
    BELT模型：用于处理长文本分类任务的神经网络模型
    
    该模型将长文本分割成多个重叠的块（chunks），分别编码后进行池化聚合，
    最终输出分类结果。支持BERT等预训练模型作为编码器。
    """
    def __init__(
        self,
        model_name: str,  # 预训练模型名称，如 'bert-base-uncased'
        num_labels: int,  # 分类标签数量
        chunk_size: int = 128,  # 每个块的大小（token数量，不包括特殊token）
        stride: int = 64,  # 滑动窗口的步长
        minimal_chunk_length: int = 50,  # 最小块长度
        pooling_strategy: str = "mean",  # 池化策略：'mean' 或 'max'
        maximal_text_length: Optional[int] = None,  # 最大文本长度限制（可选）
        use_4bit: bool = False,  # 是否使用4bit量化
        lora_cfg: Optional[Dict[str, int]] = None,  # LoRA配置参数（可选）
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

        hidden = self.model.config.hidden_size  # 预训练模型的隐藏层维度
        self.classifier = nn.Linear(hidden, num_labels)  # 线性分类器：(hidden_size,) -> (num_labels,)

        if pooling_strategy not in {"mean", "max"}:
            raise ValueError("Unknown pooling strategy!")

        self.chunk_size = chunk_size
        self.stride = stride
        self.minimal_chunk_length = minimal_chunk_length
        self.pooling_strategy = pooling_strategy
        self.maximal_text_length = maximal_text_length

    def forward(
        self,
        input_ids: List[Tensor],  # 输入token ID列表，长度为batch_size，每个元素shape为(num_chunks_i, 512)
        attention_mask: List[Tensor],  # 注意力掩码列表，长度为batch_size，每个元素shape为(num_chunks_i, 512)
        labels: Optional[Tensor] = None,  # 标签张量，shape为(batch_size,) 或 (batch_size, num_labels)
    ) -> BeltOutputs:
        """
        前向传播函数
        
        Args:
            input_ids: 输入token ID列表
                - 长度为batch_size的列表
                - 每个元素为Tensor，shape为(num_chunks_i, 512)
                - num_chunks_i为第i个样本的块数量，因文本长度而异
            attention_mask: 注意力掩码列表
                - 长度为batch_size的列表  
                - 每个元素为Tensor，shape为(num_chunks_i, 512)
                - 与input_ids对应
            labels: 标签张量（可选）
                - shape为(batch_size,) 用于二分类
                - 或shape为(batch_size, num_labels) 用于多标签分类
                
        Returns:
            BeltOutputs: 包含logits和loss的输出对象
                - logits: shape为(batch_size, num_labels)
                - loss: 如果提供labels则计算损失，否则为None
        """
        # 记录当前batch中每个样本的块数量
        # number_of_chunks: List[int], 长度为batch_size，第i个元素为样本i的块数量
        number_of_chunks = [len(x) for x in input_ids]
        
        # 将所有块拼接成一个大的批次
        # combined_ids: shape为(total_chunks, 512)，其中total_chunks = sum(number_of_chunks)
        combined_ids = torch.cat([t.to(self.classifier.weight.device) for t in input_ids], dim=0)
        # combined_mask: shape为(total_chunks, 512)
        combined_mask = torch.cat([t.to(self.classifier.weight.device) for t in attention_mask], dim=0)

        # outputs.last_hidden_state: shape为(total_chunks, 512, hidden_size)
        outputs = self.model(input_ids=combined_ids, attention_mask=combined_mask)
        
        # 使用CLS token（第0个位置）的表示进行分类
        # logits: shape为(total_chunks, num_labels)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        
        # 将logits按batch内样本重新分组
        # logits_split: List[Tensor]，长度为batch_size
        # 第i个元素shape为(number_of_chunks[i], num_labels)
        logits_split = logits.split(number_of_chunks, dim=0)
        
        # 对每个样本的多个块进行池化
        if self.pooling_strategy == "mean":
            # 平均池化：对每个样本的所有块求平均
            # pooled_logits: shape为(batch_size, num_labels)
            pooled_logits = torch.stack([torch.mean(x, dim=0) for x in logits_split])
        else:
            # 最大池化：对每个样本的所有块取最大值
            # pooled_logits: shape为(batch_size, num_labels)
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
