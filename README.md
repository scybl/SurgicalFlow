# SurgicalFlow

[English](README_en.md)

SurgicalFlow 是一个基于 PyTorch 的手术流程预测项目，面向腹腔镜胆囊切除术视频数据。项目将手术帧序列、阶段标注和器械标注组织成可训练的数据管线，用于阶段识别、剩余时间回归、未来阶段时间线预测和器械使用识别。

项目默认兼容 Cholec80 风格目录结构；仓库不分发原始视频数据，轻量检查和结果摘要可以在无数据环境下运行。

## 功能说明

| 模块 | 功能 |
| --- | --- |
| 数据读取 | 读取手术帧、阶段标注、器械标注，并生成滑动窗口序列样本 |
| 主干模型 | 使用 CNN 或 CNN-LSTM 预测当前阶段和当前阶段剩余时间 |
| 层级优化 | 使用阶段类别平衡、粗阶段组 loss 和阶段顺序距离惩罚 |
| 时间线预测 | 根据当前阶段、剩余比例和阶段先验预测未来阶段边界 |
| 时间线加权 | 对当前阶段和近未来阶段边界赋予更高训练权重 |
| 器械识别 | 使用多标签输出头预测手术器械存在情况，并支持正样本权重 |
| 可复现检查 | 生成模型摘要、结构检查结果和 README 可见输出 |

## 结果展示

| 项目 | 结果 |
| --- | ---: |
| 阶段类别 | 7 |
| 粗阶段组 | 4 |
| 器械标签 | 7 |
| 默认序列长度 | 16 frames |
| 默认步长 | 8 frames |
| 默认优化策略 | 阶段类别平衡、粗阶段组 loss、阶段顺序距离惩罚、时间线 horizon 加权、器械正样本加权 |
| `TaskA_CNN` 参数量 | 423,433 |
| `TaskA_CNN_LSTM` 参数量 | 949,769 |
| `FutureTimelineModel` 参数量 | 19,591 |
| `ToolPredictionModel` 参数量 | 17,799 |

结果文件：

- `docs/results/project_summary.md`
- `docs/results/model_summary.csv`
- `docs/results/project_summary.json`
- `docs/results/structure_check.txt`

模型摘要示例：

```csv
model,task,input_shape,output,parameters
TaskA_CNN,phase classification + remaining-time regression,"[batch, seq, 3, height, width]","phase logits, remaining-time ratio",423433
TaskA_CNN_LSTM,temporal phase classification + remaining-time regression,"[batch, seq, 3, height, width]","phase logits, remaining-time ratio",949769
```

![训练流程](picture/train_pipeline.png)

![剩余时间对比](picture/compare.jpg)

## 快速上手

配置环境并运行轻量检查：

```bash
bash scripts/setup_env.sh
bash scripts/check_project.sh
```

复用已有 conda 环境：

```bash
conda run -n codex_python bash scripts/check_project.sh
```

生成 README 对应结果文件：

```bash
make results
```

训练示例：

```bash
python train_backbone.py --name backbone_cnn --epochs 25 --model cnn --data_root data/cholec80
python train_taskA_out_head.py --name timeline_head --epochs 20 --data_root data/cholec80
python train_taskB_out_head.py --name tool_head --epochs 20 --data_root data/cholec80
```

默认训练会启用层级权重；如需退回扁平目标，可使用：

```bash
python train_backbone.py --name flat_cnn --epochs 25 --model cnn --data_root data/cholec80 --disable_class_balance --phase_group_loss_weight 0 --phase_order_loss_weight 0
python train_taskA_out_head.py --name flat_timeline --epochs 20 --data_root data/cholec80 --timeline_loss_weighting uniform
python train_taskB_out_head.py --name flat_tool --epochs 20 --data_root data/cholec80 --disable_tool_class_balance
```

## 环境要求

- Python 3.10+
- PyTorch
- 依赖见 `requirements.txt`

## 数据说明

- 仓库不包含 Cholec80 数据集。
- Cholec80 官方获取入口：[CAMMA-public/TF-Cholec80](https://github.com/CAMMA-public/TF-Cholec80)；官方脚本下载归档为 `https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz`。
- 默认数据路径为 `data/cholec80`。
- 官方说明中数据下载约需 166 GB 可用空间，解压后约 85.2 GB。
- 完整训练和评估需要整理为 `data/cholec80/frames/`、`data/cholec80/phase_annotations/` 和 `data/cholec80/tool_annotations/`。
- `picture/compare.jpg` 是已有实验记录图；提供本地数据和检查点后，可通过 `general_compare_diagram.py` 重新生成对比图。

## 目录结构

```text
model_backbone.py          CNN 和 CNN-LSTM 主干
model_out_head.py          时间线和器械输出头
workflow_schema.py         阶段、阶段组和器械标签定义
workflow_losses.py         层级 loss 和权重工具
taskA_data_loader.py       阶段/时间数据读取
taskB_data_loader.py       阶段/时间/器械数据读取
train_*.py                 训练脚本
test_*.py                  模型评估脚本
picture/                   方法图和示例结果图
docs/results/              可复现结果摘要
tests/                     轻量测试
scripts/                   环境配置、检查和结果生成脚本
```

## 测试

```bash
pytest tests/ -q
make test
```
