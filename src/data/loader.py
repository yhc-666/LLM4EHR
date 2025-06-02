from __future__ import annotations

import pickle
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset


class MIMICDataset(Dataset):
    """Dataset for MIMIC IHM/Pheno tasks."""

    def __init__(self, pkl_path: str, task: str) -> None:
        self.task = task.lower()
        with open(pkl_path, "rb") as f:
            self.data: List[Dict[str, Any]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        texts = item.get("text_data", [])
        times = item.get("text_time", list(range(len(texts))))
        order = np.argsort(times)
        texts_sorted = [texts[i] for i in order]
        if self.task == "ihm":
            label = int(item["label"])
        else:
            label = np.array(item["label"], dtype=np.float32)
        return {"text_list": texts_sorted, "label": label}
