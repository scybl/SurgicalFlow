# SurgicalFlow

[English](README.md)

SurgicalFlow 实现了一个基于 PyTorch 的时序手术流程预测流程，面向 Cholec80 格式的腹腔镜胆囊切除术数据。代码比较了基于帧特征聚合的 CNN 基线与 CNN-LSTM 时序模型，并评估输入时间窗口长度对流程预测稳定性的影响。

整体流程覆盖三个相互关联的目标：

- 当前手术阶段分类。
- 当前阶段剩余时间回归。
- 基于阶段与时间上下文的未来阶段时间线预测和器械使用预测。

## 核心功能

- 支持 Cholec80 帧数据、阶段标注和器械标注读取，默认使用 40/10/30 个视频的训练、验证和测试划分。
- 提供 CNN 与 CNN-LSTM 主干模型，并使用多任务损失联合优化阶段分类和归一化剩余时间回归。
- 提供独立输出头，用于未来阶段时间线估计和多标签器械识别。
- 训练与测试产物包含配置、日志、检查点、曲线图、JSON 指标和可选的可视化数据。
- 所有训练、测试和对比流程均通过命令行入口运行。

## 快速上手索引

| 目标 | 命令 |
| --- | --- |
| 代码结构检查 | `bash scripts/check_project.sh` |
| 复用共享 conda 环境 | `conda run -n codex_python bash scripts/check_project.sh` |
| 运行轻量测试 | `conda run -n codex_python pytest tests/ -q` |
| 检查阶段转移模式 | `python checkdata.py --phase_dir data/cholec80/phase_annotations --output outputs/phase_transition_patterns.png` |
| 训练 backbone | `python train_backbone.py --name backbone_cnn --epochs 25 --model cnn` |

## 方法概述

主干模型接收由滑动窗口采样得到的视频帧序列。CNN 负责提取每帧视觉特征。CNN 基线对时间维度进行平均聚合，CNN-LSTM 模型则使用最后一个 LSTM 状态作为时序表示。共享表示同时用于阶段分类和当前阶段剩余时间回归。

![训练流程](picture/train_pipeline.png)

输出头使用预测得到的阶段和剩余时间上下文，进一步估计未来阶段边界和器械使用情况。这样的设计将视觉表示学习与轻量结构化预测头分开。

## 仓库结构

```text
.
|-- README.md
|-- README.zh-CN.md
|-- requirements.txt
|-- checkdata.py                 # 阶段转移模式分析
|-- general_compare_diagram.py   # 剩余时间预测对比图
|-- model_backbone.py            # CNN 与 CNN-LSTM 主干模型
|-- model_out_head.py            # 时间线与器械输出头
|-- taskA_data_loader.py         # 阶段/时间数据读取
|-- taskB_data_loader.py         # 阶段/时间/器械数据读取
|-- test_backbone.py             # 主干模型评估
|-- test_taskA_out_head.py       # 时间线输出头流水线评估
|-- test_taskB_out_head.py       # 器械输出头流水线评估
|-- train_backbone.py            # 主干模型训练
|-- train_taskA_out_head.py      # 时间线输出头训练
|-- train_taskB_out_head.py      # 器械输出头训练
`-- picture/
    |-- compare.jpg
    `-- train_pipeline.png
```

## 环境配置

推荐使用 Python 3.10。

一键结构检查：

```bash
bash scripts/check_project.sh
```

如果已经有共享 conda 环境，可以直接复用：

```bash
conda run -n codex_python bash scripts/check_project.sh
```

手动安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果默认包源中的 PyTorch 版本与本机 CUDA 或 CPU 运行环境不匹配，请安装与目标机器对应的 PyTorch 构建版本。

## 数据集

代码默认读取本地准备好的 Cholec80 数据。本仓库不包含数据集文件。

使用 CAMMA TF-Cholec80 脚本准备数据集：

```bash
git clone https://github.com/CAMMA-public/TF-Cholec80.git
cd TF-Cholec80
python prepare.py --data_rootdir /absolute/path/to/datasets
```

`prepare.py` 会将 Cholec80 下载并解压到 `/absolute/path/to/datasets/cholec80`。官方脚本还支持使用 `--verify_checksum` 校验下载归档，或使用 `--keep_archive` 在解压后保留下载归档。运行下载和解压步骤前，请确认本地有足够磁盘空间。

准备完成后，可以在运行训练或测试脚本时显式传入解压后的目录：

```bash
python train_backbone.py \
  --name backbone_cnn \
  --epochs 25 \
  --model cnn \
  --data_root /absolute/path/to/datasets/cholec80
```

也可以将数据目录链接或复制到本仓库默认读取的位置：

```bash
mkdir -p data
ln -s /absolute/path/to/datasets/cholec80 data/cholec80
```

期望目录结构如下：

```text
data/cholec80/
|-- frames/
|   |-- video01/
|   |-- video02/
|   `-- ...
|-- phase_annotations/
|   |-- video01-phase.txt
|   |-- video02-phase.txt
|   `-- ...
`-- tool_annotations/
    |-- video01-tool.txt
    |-- video02-tool.txt
    `-- ...
```

如果使用 CAMMA TF-Cholec80 准备脚本，可将处理后的数据放在 `data/cholec80` 下，或通过 `--data_root` 指定自定义路径。

检查阶段转移模式：

```bash
python checkdata.py \
  --phase_dir data/cholec80/phase_annotations \
  --output outputs/phase_transition_patterns.png
```

## 训练

训练 CNN 主干模型：

```bash
python train_backbone.py \
  --name backbone_cnn \
  --epochs 25 \
  --model cnn
```

使用不同时间窗口训练 CNN-LSTM 主干模型：

```bash
python train_backbone.py \
  --name backbone_cnn_lstm_16 \
  --epochs 25 \
  --model cnn_lstm \
  --seq_len 16 \
  --stride 8

python train_backbone.py \
  --name backbone_cnn_lstm_32 \
  --epochs 25 \
  --model cnn_lstm \
  --seq_len 32 \
  --stride 16
```

训练结构化输出头：

```bash
python train_taskA_out_head.py \
  --name timeline_head_16 \
  --epochs 25 \
  --seq_len 16 \
  --stride 8

python train_taskB_out_head.py \
  --name tool_head_16 \
  --epochs 25 \
  --seq_len 16 \
  --stride 8
```

训练产物写入 `checkpoints/<experiment_name>/`，包括：

- `best.pth`
- `config.json`
- `train.log`
- 主干模型的 `training_curve.png`
- 输出头的 `loss_curve.png`

## 评估

评估已训练的主干模型：

```bash
python test_backbone.py \
  --name backbone_cnn_lstm_16 \
  --model cnn_lstm \
  --seq_len 16 \
  --stride 8
```

使用已训练主干和时间线输出头评估完整时间线流程：

```bash
python test_taskA_out_head.py \
  --backbone_name backbone_cnn_lstm_16 \
  --backbone_model cnn_lstm \
  --head_name timeline_head_16 \
  --seq_len 16 \
  --stride 8
```

评估器械识别流程：

```bash
python test_taskB_out_head.py \
  --backbone_name backbone_cnn_lstm_16 \
  --backbone_model cnn_lstm \
  --head_name tool_head_16 \
  --seq_len 16 \
  --stride 8
```

评估结果会写入对应检查点目录中的 `test.log` 和 `test_result.json`。时间线评估还会写入 `future_timeline_data.npz`。

## 指标

- 主干模型评估输出阶段准确率、以秒为单位的剩余时间 MAE，以及当前阶段剩余时间预测的 R2。
- 时间线输出头评估输出阶段准确率、开始时间 MAE、结束时间 MAE，以及有效未来时间线点上的 R2。
- 器械输出头评估输出器械准确率、micro-F1 和 macro-F1，用于多标签器械使用预测。

## 时间窗口对比

`general_compare_diagram.py` 会在同一测试子集上比较多个已训练主干模型，并绘制剩余时间预测与真实值的对比图。如果实验名称与默认值不同，请修改文件顶部的 `MODELS` 列表。

```bash
python general_compare_diagram.py \
  --data_root data/cholec80 \
  --save_dir checkpoints \
  --output outputs/remaining_time_comparison.png
```

示例对比图位于 `picture/compare.jpg`。

![剩余时间对比](picture/compare.jpg)

## 可复现性说明

- 脚本通过 `--seed` 设置 Python、NumPy 和 PyTorch 随机种子。
- 当 CUDA 不可用时，脚本会自动回退到 CPU。
- 代码通过每 25 条标注读取一次，将视频帧按 1 FPS 采样。
- 默认划分在视频文件夹按字典序排序后，使用第 `1-40` 个视频训练，第 `41-50` 个视频验证，第 `51-80` 个视频测试。
- `data/`、`checkpoints/` 和生成输出目录不纳入版本控制。

## 主要参数

| 参数 | 使用位置 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--name` | 训练脚本、`test_backbone.py` | 必填 | 实验名称与检查点子目录 |
| `--epochs` | 训练脚本 | 必填 | 训练轮数 |
| `--model` | 主干训练/测试 | 必填 | `cnn` 或 `cnn_lstm` |
| `--backbone_name` | 输出头测试 | 必填 | 主干模型检查点目录 |
| `--head_name` | 输出头测试 | 必填 | 输出头检查点目录 |
| `--backbone_model` | 输出头测试 | 必填 | 检查点对应的主干架构 |
| `--data_root` | 所有训练/测试脚本 | `data/cholec80` | 数据集根目录 |
| `--batch_size` | 所有训练/测试脚本 | `16` | 批大小 |
| `--lr` | 训练脚本 | `1e-4` | Adam 学习率 |
| `--seq_len` | 序列脚本 | `16` | 每个时间窗口的帧数 |
| `--stride` | 序列脚本 | `8` | 滑动窗口步长 |
| `--num_workers` | 数据加载 | `8` | DataLoader 工作进程数 |
| `--save_dir` | 主干训练/测试、对比图脚本 | `checkpoints` | 产物根目录 |
| `--device` | 所有训练/测试脚本 | `cuda` | 请求使用的设备 |
| `--seed` | 所有训练/测试脚本 | `42` | 随机种子 |
