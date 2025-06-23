from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class LSTMOutputs:
    """Output container for :class:`LSTMClassifier`."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class LSTMClassifier(nn.Module):
    """Simple LSTM-based classifier for regularised time-series.

    Parameters
    ----------
    input_dim: int
        Number of features at each time step.
    hidden_dim: int
        Hidden size of the LSTM.
    num_layers: int
        Number of stacked LSTM layers.
    num_labels: int
        Dimension of the prediction output.
    dropout: float, default 0.1
        Dropout applied after the LSTM layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_labels)

        # ``tokenizer`` attribute for compatibility with existing pipeline.
        self.tokenizer = None

    def forward(
        self, reg_ts: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> LSTMOutputs:
        """Forward pass of the classifier.

        Parameters
        ----------
        reg_ts: Tensor
            Regularised time series of shape ``(B, T, F)``.
        labels: Tensor, optional
            Target labels ``(B,)`` or ``(B, num_labels)``.

        Returns
        -------
        :class:`LSTMOutputs`
            ``logits`` has shape ``(B, num_labels)``.
        """
        out, _ = self.lstm(reg_ts)
        # final hidden state corresponds to the last time step
        last = out[:, -1]
        logits = self.fc(self.dropout(last))

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            if logits.size(1) == 1 or labels.ndim == 1:
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                loss = loss_fn(logits, labels.float())
        return LSTMOutputs(logits=logits, loss=loss)
