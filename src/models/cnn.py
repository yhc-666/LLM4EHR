from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class CNNOutputs:
    """Output container for :class:`CNNClassifier`."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class CNNClassifier(nn.Module):
    """1D CNN baseline for regularised time-series.

    Parameters
    ----------
    input_dim: int
        Number of features at each time step (channels).
    num_filters: int
        Number of convolutional filters.
    kernel_size: int
        Size of the temporal convolution kernel.
    num_labels: int
        Dimension of the prediction output.
    dropout: float, default 0.1
        Dropout applied before the final linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int,
        kernel_size: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_labels)

        # ``tokenizer`` attribute for compatibility with existing pipeline
        self.tokenizer = None

    def forward(
        self, reg_ts: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> CNNOutputs:
        """Forward pass of the CNN classifier.

        Parameters
        ----------
        reg_ts: Tensor
            Regularised time series of shape ``(B, T, F)``.
        labels: Tensor, optional
            Target labels ``(B,)`` or ``(B, num_labels)``.

        Returns
        -------
        :class:`CNNOutputs`
            ``logits`` has shape ``(B, num_labels)``.
        """
        x = reg_ts.permute(0, 2, 1)  # (B, F, T)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(self.dropout(x))

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            if logits.size(1) == 1 or labels.ndim == 1:
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                loss = loss_fn(logits, labels.float())
        return CNNOutputs(logits=logits, loss=loss)
