from __future__ import annotations

import pickle
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset


class MIMICDataset(Dataset):
    """Dataset for MIMIC IHM/Pheno tasks.

    Parameters
    ----------
    pkl_path: str
        Path to the pickle file containing the dataset.
    task: str
        Either ``"ihm"`` or ``"pheno"``.
    model_type: str, default "llama"
        Which model this dataset will feed. ``"timellm"`` additionally returns
        the regularized time series.
    """

    def __init__(self, pkl_path: str, task: str, model_type: str = "llama") -> None:
        self.task = task.lower()
        self.model_type = model_type.lower()
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
        if self.model_type == "timellm":
            out["reg_ts"] = item["reg_ts"].astype(np.float32)
        return out



if __name__ == "__main__":
    pkl_path = "/home/ubuntu/hcy50662/output_mimic3/pheno/test_p2x_data.pkl"  # 替换为实际的pkl文件路径
    task = "pheno"  # 或者 "pheno"，根据你的任务类型
    
    try:
        # 创建dataset实例
        dataset = MIMICDataset(pkl_path, task)
        print(f"数据集大小: {len(dataset)}")
        
        # 打印前几个样本
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n--- 样本 {i} ---")
            print(f"文本数量: {len(sample['text_list'])}")
            print(f"标签: {sample['label']}")
            print(f"标签长度: {len(sample['label'])}")
            print(f"标签类型: {type(sample['label'])}")
            if sample['text_list']:
                print(f"第一段文本预览: {sample['text_list'][0][:100]}...")
                
    except FileNotFoundError:
        print(f"错误: 找不到文件 {pkl_path}")
        print("请确认pkl文件路径是否正确")
    except Exception as e:
        print(f"加载数据时出错: {e}")
