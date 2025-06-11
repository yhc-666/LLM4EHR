from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, GPT2Model, GPT2Config
from einops import rearrange


class RevIn(nn.Module):
    """Simple reversible instance normalization used in GPT4MTS."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.weight + self.bias
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps * self.eps)
            x = x * self.std + self.mean
            return x
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")


@dataclass
class GPT4MTSOutput:
    """Output of :class:`GPT4MTS`."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class HierAggregationHead(nn.Module):
    """Hierarchical aggregation classifier head.

    Parameters
    ----------
    d_model: int
        Dimension of GPT hidden states.
    num_classes: int
        Number of prediction classes.
    max_channels: int, default 17
        Maximum number of time-series channels.
    d_mid: int, default 256
        Hidden dimension inside the head.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        max_channels: int = 17,
        d_mid: int = 256,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_mid, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_mid))
        self.patch_attn = nn.TransformerEncoderLayer(d_mid, nhead=4, batch_first=True)
        self.var_emb = nn.Parameter(torch.randn(max_channels, d_mid))
        self.chan_qk = nn.Linear(d_mid, 2 * d_mid, bias=False)
        self.norm = nn.LayerNorm(d_mid)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(d_mid, num_classes)
        self.d_mid = d_mid

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Aggregate patch embeddings hierarchically.

        Parameters
        ----------
        h: Tensor of shape ``(B, C, N, d_model)``

        Returns
        -------
        Tensor of shape ``(B, num_classes)``.
        """

        B, C, N, _ = h.shape
        h_patch = self.proj(h)  # (B, C, N, d_mid)
        h_patch = h_patch.view(B * C, N, self.d_mid)
        cls = self.cls_token.expand(B * C, -1, -1)
        h_patch = torch.cat([cls, h_patch], dim=1)
        h_chan = self.patch_attn(h_patch)[:, 0, :]
        h_chan = h_chan.view(B, C, self.d_mid)
        h_chan = h_chan + self.var_emb[:C]
        qk = self.chan_qk(h_chan)
        q, k = qk.chunk(2, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (self.d_mid ** 0.5)
        attn = attn.softmax(dim=-1)
        g = (attn @ h_chan).mean(dim=1)
        logits = self.fc(self.drop(self.norm(g)))
        return logits


class GPT4MTS(nn.Module):
    """PyTorch implementation of GPT4MTS for classification tasks.

    Parameters
    ----------
    gpt_model: str
        Name of the GPT2 model to load.
    num_labels: int
        Number of prediction labels.
    seq_len: int
        Length of the regularized time series.
    patch_size: int
        Patch length when patchifying the time series and text embeddings.
    stride: int
        Stride when patchifying.
    gpt_layers: int, default 6
        Number of GPT2 layers to keep.
    d_model: int, default 768
        Hidden dimension fed into GPT2.
    freeze: bool, default False
        If ``True``, freeze GPT2 parameters except layernorms and positional embeddings.
    pretrain: bool, default True
        If ``True``, load pretrained GPT2 weights, otherwise initialize from scratch.
    revin: bool, default False
        Apply reversible instance normalization on the input time series.
    """

    def __init__(
        self,
        gpt_model: str,
        num_labels: int,
        seq_len: int,
        patch_size: int = 8,
        stride: int = 4,
        gpt_layers: int = 6,
        d_model: int = 768,
        freeze: bool = False,
        pretrain: bool = True,
        revin: bool = False,
        classifier_head: str = "linear",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.patch_num = (seq_len - patch_size) // stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1  # extra patch due to replication padding
        self.d_model = d_model
        self.revin = revin
        self.num_labels = num_labels
        self.classifier_head_type = classifier_head

        if pretrain:
            gpt2 = GPT2Model.from_pretrained(gpt_model)
        else:
            gpt2 = GPT2Model(GPT2Config())
        gpt2.h = gpt2.h[:gpt_layers]
        self.gpt2 = gpt2

        if freeze and pretrain:
            for name, param in self.gpt2.named_parameters():
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.in_layer = nn.Linear(patch_size, d_model)

        # text encoder
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.prompt_layer = nn.Linear(self.text_encoder.config.hidden_size, d_model)
        self.relu = nn.ReLU()

        if self.classifier_head_type == "linear":
            self.classifier_head = nn.LazyLinear(num_labels)
        elif self.classifier_head_type == "hier":
            self.classifier_head = HierAggregationHead(d_model, num_labels)
        else:
            raise ValueError("unknown classifier_head")
        self.rev_in = RevIn(num_features=1)

    def get_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify time series into overlapping segments.

        Parameters
        ----------
        x: Tensor of shape ``(B, L, C)``

        Returns
        -------
        Tensor of shape ``(B*C, N, patch_size)`` where ``N`` is number of patches.
        """
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        return x

    def patch_summary(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        """Patchify text-note embeddings following GPT4MTS."""
        device = embeds[0].device
        padded = nn.utils.rnn.pad_sequence(embeds, batch_first=True)
        if padded.size(1) < self.patch_size:
            extra = self.patch_size - padded.size(1)
            last = padded[:, -1:, :].repeat(1, extra, 1)
            padded = torch.cat([padded, last], dim=1)
        summary = rearrange(padded, 'b l m -> b m l')
        summary = self.padding_patch_layer(summary)
        summary = summary.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        summary = summary.mean(dim=-1)
        summary = rearrange(summary, 'b m l -> b l m')
        return summary.to(device)

    def encode_notes(
        self, notes: List[Dict[str, torch.Tensor]], device: torch.device
    ) -> List[torch.Tensor]:
        """Encode a batch of notes with the text encoder."""
        embeds = []
        for enc in notes:
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.text_encoder(**enc)
            embeds.append(out.last_hidden_state[:, 0, :])
        return embeds

    def get_emb(self, x: torch.Tensor, tokens: torch.Tensor | None = None) -> torch.Tensor:
        if tokens is None:
            return self.gpt2(inputs_embeds=x).last_hidden_state
        a, b, _ = x.shape
        prompt_x = self.relu(self.prompt_layer(tokens))
        x_all = torch.cat([prompt_x, x], dim=1)
        out = self.gpt2(inputs_embeds=x_all).last_hidden_state
        return out[:, -b:, :]


    def forward(
        self,
        reg_ts: torch.Tensor,
        summary_tokens: List[Dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
    ) -> GPT4MTSOutput:
        """Forward pass.

        Parameters
        ----------
        reg_ts: Tensor
            Regularized time series of shape ``(B, L, C)``.
        summary_tokens: List[Dict[str, Tensor]]
            Tokenized notes for each sample.
        labels: Optional tensor
            Target labels of shape ``(B,)`` or ``(B, num_labels)``.
        """
        B, L, C = reg_ts.size()
        device = reg_ts.device
        if self.revin:
            reg_ts = self.rev_in(reg_ts, "norm")
        else:
            mean = reg_ts.mean(1, keepdim=True).detach()
            std = torch.sqrt(reg_ts.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
            reg_ts = (reg_ts - mean) / std

        x = self.get_patch(reg_ts)
        x = self.in_layer(x)

        note_embeds = self.encode_notes(summary_tokens, device)
        summary_prompt = self.patch_summary(note_embeds)
        summary_prompt = summary_prompt.repeat_interleave(C, dim=0)
        h = self.get_emb(x, summary_prompt)
        h = h.reshape(B, C, self.patch_num, self.d_model)
        if self.classifier_head_type == "linear":
            logits = self.classifier_head(h.reshape(B, -1))
        else:
            logits = self.classifier_head(h)

        if self.revin:
            pass  # outputs are classification logits, no denorm needed
        else:
            pass

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            if logits.size(1) == 1 or labels.ndim == 1:
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                loss = loss_fn(logits, labels.float())
        return GPT4MTSOutput(logits=logits, loss=loss)
