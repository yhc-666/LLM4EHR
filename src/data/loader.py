from __future__ import annotations

import pickle
from typing import Any, Dict, List

from ..utils import BaseConfig

import numpy as np
from torch.utils.data import Dataset


class MIMICDataset(Dataset):
    """Dataset for MIMIC IHM/Pheno tasks.

    Parameters
    ----------
    cfg: :class:`~src.utils.BaseConfig`
        Experiment configuration object. Should contain ``<split>_pkl`` paths,
        ``task`` and ``model_type`` fields.
        split: str, optional
        Dataset split to load. Can be ``"train"``, ``"val"`` or ``"test"``.
        Defaults to ``"train"``.
    Possible model types:
        - llama
        - timellm
        - gpt4mts
        - clinicalbigbird
        - clinicalbert
    Possible tasks:
        - ihm
        - pheno
    """

    def __init__(self, cfg: BaseConfig, split: str = "train") -> None:
        self.cfg = cfg
        self.task = cfg.task.lower()
        self.model_type = cfg.model_type.lower()

        path_attr = f"{split}_pkl"
        pkl_path = getattr(cfg, path_attr)
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
            label = np.array(item["label"][1:], dtype=np.float32)
        out = {"text_list": texts_sorted, "label": label}
        if self.model_type in {"timellm", "gpt4mts"}:
            out["reg_ts"] = item["reg_ts"][:, :17].astype(np.float32)
        elif self.model_type == "lstm":
            out["reg_ts"] = item["reg_ts"].astype(np.float32)
        elif self.model_type == "clinicalbigbird":
            pass  # same as text-only models
        return out



if __name__ == "__main__":
    from pathlib import Path
    from ..utils import parse_config_yaml

    config_path = Path(__file__).resolve().parents[2] / "config/pheno_timellm.yaml"
    cfg = parse_config_yaml(str(config_path))

    try:
        dataset = MIMICDataset(cfg, split="test")
        print(f"数据集大小: {len(dataset)}")
        
        # 打印前几个样本
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n--- 样本 {i} ---")
            print(f"文本数量: {len(sample['text_list'])}")
            print(f"标签: {sample['label']}")
            print(f"标签长度: {len(sample['label'])}")
            print(f"reg_ts长度: {sample['reg_ts'].shape}")
            print(f"reg_ts: {sample['reg_ts']}")
            print(f"标签类型: {type(sample['label'])}")
            if sample['text_list']:
                print(f"第一段文本预览: {sample['text_list'][0][:100]}...")
                
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}")
        print("请确认pkl文件路径是否正确")
