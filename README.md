# SurgicalFlow

[English](README_en.md)

SurgicalFlow 是一个基于 PyTorch 的手术流程预测项目，面向 Cholec80 格式的腹腔镜胆囊切除术数据。项目包含 CNN、CNN-LSTM、时间线预测头和器械识别头。

![SurgicalFlow 时序建模预览](docs/images/surgical-flow-preview.svg)

## 功能说明

- 读取手术帧、阶段标注和器械标注。
- 训练 CNN 或 CNN-LSTM 主干模型。
- 预测当前阶段、当前阶段剩余时间、未来阶段时间线和器械使用情况。
- 输出训练日志、曲线图、检查点和 JSON 指标。

## 结果展示

| 项目 | 内容 |
| --- | --- |
| 任务 | 阶段分类、剩余时间回归、时间线预测、器械识别 |
| 模型 | CNN baseline、CNN-LSTM |
| 方法图 | `picture/train_pipeline.png` |
| 对比图 | `picture/compare.jpg` |

![训练流程](picture/train_pipeline.png)

![剩余时间对比](picture/compare.jpg)

## 快速上手

检查项目结构：

```bash
bash scripts/check_project.sh
```

复用已有 conda 环境：

```bash
conda run -n codex_python bash scripts/check_project.sh
```

训练示例：

```bash
python train_backbone.py --name backbone_cnn --epochs 25 --model cnn --data_root data/cholec80
```

## 环境要求

- Python 3.10+
- PyTorch
- 依赖见 `requirements.txt`

## 数据说明

- 仓库不包含 Cholec80 数据集。
- 默认数据路径为 `data/cholec80`。
- 完整训练和评估需要本地准备 `frames/`、`phase_annotations/` 和 `tool_annotations/`。

## 目录结构

```text
model_backbone.py          CNN 和 CNN-LSTM 主干
model_out_head.py          时间线和器械输出头
taskA_data_loader.py       阶段/时间数据读取
taskB_data_loader.py       阶段/时间/器械数据读取
train_*.py                 训练脚本
test_*.py                  评估脚本
picture/                   方法图和示例结果图
tests/                     轻量测试
scripts/                   检查脚本
```

## 测试

```bash
pytest tests/ -q
```
