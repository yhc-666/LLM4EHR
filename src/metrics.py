from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from scipy.special import expit


def binary_metrics(pred_logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute metrics for binary classification."""
    probs = expit(pred_logits.squeeze())
    preds = (probs >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(labels, probs),
        "AUPRC": average_precision_score(labels, probs),
        "F1": f1_score(labels, preds),
        "ACC": accuracy_score(labels, preds),
    }


def multilabel_metrics(pred_logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute metrics for multi-label classification."""
    probs = expit(pred_logits)
    pred_bin = (probs >= 0.5).astype(int)
    return {
        "macro_AUROC": roc_auc_score(labels, probs, average="macro"),
        "macro_AUPRC": average_precision_score(labels, probs, average="macro"),
        "micro_F1": f1_score(labels, pred_bin, average="micro"),
        "macro_F1": f1_score(labels, pred_bin, average="macro"),
    }

