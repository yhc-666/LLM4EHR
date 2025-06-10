from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


@dataclass
class TimeLLMOutputs:
    """Output of ``TimeLLM`` forward pass."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class TokenEmbedding(nn.Module):
    """Two-layer MLP token embedding used for patchified time series.

    Args:
        c_in: number of channels in each patch (``patch_len`` × ``num_vars``)
        d_model: embedding dimension of the patch tokens

    Input shape: ``(B, L, c_in)`` where ``L`` is number of patches.
    Output shape: ``(B, L, d_model)``.
    """

    def __init__(self, c_in: int | None, d_model: int) -> None:
        super().__init__()
        if c_in is None:
            self.fc1 = nn.LazyLinear(d_model)
        else:
            self.fc1 = nn.Linear(c_in, d_model)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(x)


class ReplicationPad1d(nn.Module):
    """Pad the last timestep by replication used before patchifying."""

    def __init__(self, padding: Tuple[int, int]) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        replicate = x[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        return torch.cat([x, replicate], dim=-1)


class PatchEmbedding(nn.Module):
    """Patchify multi-variate time series and embed each patch.

    Args:
        d_model: embedding dimension after convolution
        patch_len: patch length along the time dimension
        stride: stride for unfolding
        dropout: dropout rate applied after embedding

    Input shape: ``(B, V, T)`` where ``V`` is number of variables.
    Output shape: ``(B, L, d_model)`` where ``L`` is number of patches.
    """

    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad = ReplicationPad1d((0, stride))
        # ``TokenEmbedding`` will lazily infer its input dimension during the first forward pass
        self.value_embedding = TokenEmbedding(None, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, V, T = x.size()
        x = self.pad(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, V, L, patch_len)
        L = x.size(2)
        x = x.permute(0, 2, 1, 3).reshape(B, L, V * self.patch_len)
        x = self.value_embedding(x)
        return self.dropout(x), V


class ReprogrammingLayer(nn.Module):
    """Cross attention layer for reprogramming patch embeddings into the LLM space."""

    def __init__(self, d_model: int, n_heads: int, d_llm: int) -> None:
        super().__init__()
        d_keys = d_model // n_heads
        self.query_proj = nn.Linear(d_model, d_keys * n_heads)
        self.key_proj = nn.Linear(d_llm, d_keys * n_heads)
        self.value_proj = nn.Linear(d_llm, d_keys * n_heads)
        self.out_proj = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(0.1)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Apply reprogramming cross attention.

        Args:
            target: patch embeddings of shape ``(B, L, d_model)``
            source: source token embeddings of shape ``(S, d_llm)``
        Returns:
            Tensor of shape ``(B, L, d_llm)``
        """

        B, L, _ = target.size()
        S, _ = source.size()
        H = self.n_heads
        Q = self.query_proj(target).view(B, L, H, -1)
        K = self.key_proj(source).view(S, H, -1)
        V = self.value_proj(source).view(S, H, -1)
        scores = torch.einsum("blhe,she->bhls", Q, K) / (K.size(-1) ** 0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        rep = torch.einsum("bhls,she->blhe", attn, V)
        rep = rep.reshape(B, L, -1)
        return self.out_proj(rep)


class TimeLLM(nn.Module):
    """Time-LLM model wrapper supporting time series + text inputs.

    Parameters
    ----------
    model_name: str
        Name of the pretrained language model.
    num_labels: int
        Number of prediction labels.
    use_4bit: bool, default False
        Whether to load the LLM with 4-bit weights.
    lora_cfg: Optional[Dict[str, int]]
        LoRA configuration if using parameter-efficient tuning.
    d_model: int | None
        Patch embedding dimension. ``None`` defaults to LLM hidden size.
    patch_len: int
        Length of each time-series patch.
    stride: int
        Stride when patchifying time series.
    n_heads: int
        Number of heads in the reprogramming attention.
    freeze_base_model: bool, default False
        If ``True``, freeze all parameters of the base LLM.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        use_4bit: bool = False,
        lora_cfg: Optional[Dict[str, int]] = None,
        d_model: int | None = None,
        patch_len: int = 8,
        stride: int = 8,
        n_heads: int = 8,
        freeze_base_model: bool = False,
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
            torch_dtype=torch.bfloat16,
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
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        hidden = self.model.config.hidden_size
        d_model = d_model or hidden
        self.patch_embed = PatchEmbedding(d_model, patch_len, stride, 0.1)
        self.reprogram = ReprogrammingLayer(d_model, n_heads, hidden)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_tokens = 1000

        if freeze_base_model:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reg_ts: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> TimeLLMOutputs:
        """Forward pass of Time-LLM.

        Args:
            input_ids: token ids of shape ``(B, L_text)``
            attention_mask: mask for text tokens ``(B, L_text)``
            reg_ts: regularized time series ``(B, T, F)``
            labels: optional labels ``(B,)`` or ``(B, num_labels)``
        """

        text_embeds = self.model.get_input_embeddings()(input_ids)
        ts = reg_ts.permute(0, 2, 1).contiguous()  # (B, F, T)
        ts_embed, _ = self.patch_embed(ts)
        vocab_embed = self.model.get_input_embeddings().weight[: self.num_tokens]
        ts_embed = self.reprogram(ts_embed, vocab_embed)

        patch_mask = torch.ones(ts_embed.size(0), ts_embed.size(1), device=ts_embed.device)
        concat_embeds = torch.cat([ts_embed, text_embeds], dim=1)
        concat_mask = torch.cat([patch_mask, attention_mask], dim=1)

        outputs = self.model(inputs_embeds=concat_embeds, attention_mask=concat_mask, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state
        mask = concat_mask.unsqueeze(-1)
        pooled = (last_hidden * mask).sum(1) / mask.sum(1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            if logits.size(1) == 1 or labels.ndim == 1:
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                loss = loss_fn(logits, labels.float())
        return TimeLLMOutputs(logits=logits, loss=loss)
