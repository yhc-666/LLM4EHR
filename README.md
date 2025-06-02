# LLM4EHR
LLM4EHR is a minimal-yet-extensible PyTorch codebase for benchmarking long-context LLMs on two classic ICU prediction tasks using the MIMIC-III / IV datasets:

| Task                                         | Horizon & Labels | Input Modalities (used today)                |
| -------------------------------------------- | ---------------- | -------------------------------------------- |
| **48-hour In-Hospital Mortality (IHM)**      | binary (0 / 1)   | *N clinical notes* per stay (≈ 4000 tokens) |
| **24-hour Phenotype Classification (Pheno)** | 25× multi-label  | *N clinical notes* per stay                  |


The current baseline feeds all notes into Meta-Llama-3-8B (optionally 4-bit + LoRA) and aggregates its final hidden states with masked mean pooling → linear head.

## Project Stucture
```
LLM4EHR/
├── README.md                  # 项目说明 + 快速上手
├── requirements.txt           # 依赖 (torch, transformers, accelerate …)
│
├── config/                    # 每个 YAML = 一组实验超参
│   ├── ihm.yaml               # 48 h IHM 二分类
│   └── pheno.yaml             # 24 h Pheno 多标签
│
├── src/
│   ├── data/
│   │   ├── loader.py          # 读 raw pkl ➜ torch Dataset
│   │   └── collate.py         # tokenize→pad
│   │
│   ├── models/
│   │   └── llama_mean.py      # Llama-3 8B + masked-mean pooling + 线性头 (+LoRA/4-bit)
│   │
│   ├── metrics.py             # AUPRC / AUROC / F1 / ACC 计算
│   ├── train.py               # Accelerate 驱动的纯-PyTorch 训练循环
│   ├── test.py                # 推理 + 调用 metrics 评估
│   └── utils.py               # set_seed, save_checkpoint, YAML 解析等工具函数
│
└── scripts/
    ├── run_ihm.sh             # 运行脚本
    └── run_pheno.sh
```

## Dataset Format
首先我在预处理后得到ihm/pheno 分别的下游prediction任务数据集，每个数据集被切分为train/val/test，以pkl格式存储于'DATAROOT'

1. 48h ihm prediction：对应train_p2x_data.pkl, val_p2x_data.pkl, test_p2x_data.pkl

一共14066条

| 字段名        | 数据类型            | Shape / 长度示例                  | 描述                                                         | 含义说明                                  |
| ------------- | ------------------- | --------------------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| `reg_ts`      | `numpy.ndarray`     | `(48, 34)`                        | 规则化后的多变量时间序列（每小时一次，共 48 小时）。前 17 列为数值特征（生命体征、检验指标等），后 17 列为对应的特征缺失掩码 | 结构化、均匀采样处理后的时序输入          |
| `irg_ts`      | `numpy.ndarray`     | `(len, 17)`, len的大小不固定      | 原始不规则采样的多变量时间序列，包含缺失                     | 真实测量值的时间序列输入(未经48h对齐处理) |
| `irg_ts_mask` | `numpy.ndarray`     | `(len, 17)`, len的大小不固定      | 同上                                                         | 缺失标记（1 = 存在，0 = 缺失）            |
| `ts_tt`       | `list[float]`       | `length = len`                    | 与 `irg_ts` 行对应的时间戳（单位：小时）                     | 每条时序测量的发生时间                    |
| `text_data`   | `list[str]`         | 长度不固定                        | 临床自由文本（护理/病程记录等）                              | 非结构化文本输入                          |
| `text_time`   | `list[float]`       | `length = len(text_data)`         | 文本对应的时间戳（小时）                                     | 文本事件发生时间                          |
| `label`       | `list[int]` / `int` | IHM: `()`                         | 预测标签：院内死亡（二分类 0/1）                             | 监督信号                                  |
| `name`        | `str`               | 例: 10163_episode1_timeseries.csv | 样本文件名或唯一标识                                         | 便于追溯与调试                            |

2. 24h phenotype classification

一共23163条

| 字段名        | 数据类型            | Shape / 长度示例               | 描述                                                         | 含义说明                       |
| ------------- | ------------------- | ------------------------------ | ------------------------------------------------------------ | ------------------------------ |
| `reg_ts`      | `numpy.ndarray`     | `(24, 34)`                     | 规则化后的多变量时间序列（每小时一次，共 24 小时）。前 17 列为数值特征（生命体征、检验指标等），后 17 列为对应的特征缺失掩码 | 结构化、均匀采样后的时序输入   |
| `irg_ts`      | `numpy.ndarray`     | ``(len, 17)`, len的大小不固定` | 原始不规则采样的多变量时间序列，包含缺失                     | 真实测量值的时间序列输入       |
| `irg_ts_mask` | `numpy.ndarray`     | ``(len, 17)`, len的大小不固定` | 同上                                                         | 缺失标记（1 = 存在，0 = 缺失） |
| `ts_tt`       | `list[float]`       | `length = len`                 | 与 `irg_ts` 行对应的时间戳（单位：小时）                     | 每条时序测量的发生时间         |
| `text_data`   | `list[str]`         | 长度不固定                     | 临床自由文本（护理/病程记录等）                              | 非结构化文本输入               |
| `text_time`   | `list[float]`       | ``length = len(text_data)``    | 文本对应的时间戳（小时）                                     | 文本事件发生时间               |
| `label`       | `list[int]` / `int` | PHE: `(25,)`                   | 预测标签：25 类表型多标签（0/1 × 25）                        | 监督信号                       |
| `name`        | `str`               | —                              | 样本文件名或唯一标识                                         | 便于追溯与调试                 |
