# SurgicalFlow

[中文](README.md)

SurgicalFlow is a PyTorch surgical workflow prediction project for laparoscopic cholecystectomy videos. It turns frame sequences, phase annotations, and tool annotations into trainable data pipelines for phase recognition, remaining-time regression, future phase-boundary prediction, and surgical tool recognition.

The default layout targets Cholec80 data. Cholec80 contains 80 laparoscopic cholecystectomy videos from 13 surgeons; a common prepared layout stores per-video frame sequences with 7 surgical phase labels and 7 surgical tool-presence labels. The labels are not flat in practice: phases follow a surgical workflow order, phase durations are imbalanced, and tool labels are multi-label binary targets affected by tool visibility. SurgicalFlow therefore uses sequence windows, phase-group constraints, class balancing, and timeline weighting by default.

Raw videos are not distributed with this repository; lightweight checks and result summaries run without the dataset.

## Features

| Module | Capability |
| --- | --- |
| Data loading | Reads frames, phase labels, tool labels, and builds sliding-window sequence samples |
| Backbones | Uses CNN or CNN-LSTM models for current-phase prediction and remaining-time regression |
| Hierarchical optimization | Uses phase class balancing, coarse phase-group loss, and ordinal phase-distance loss |
| Timeline prediction | Predicts future phase boundaries from current phase, remaining ratio, and phase priors |
| Timeline weighting | Gives higher training weight to the current and near-future phase boundaries |
| Tool recognition | Uses a multi-label output head for surgical tool presence with positive-class weighting |
| Reproducible checks | Generates model summaries, structure-check output, and README-visible artifacts |

## Results

| Item | Result |
| --- | ---: |
| Phase classes | 7 |
| Coarse phase groups | 4 |
| Tool labels | 7 |
| Default sequence length | 16 frames |
| Default stride | 8 frames |
| Default optimization | phase class balance, phase group loss, phase order loss, timeline horizon weighting, tool positive-class weighting |
| `TaskA_CNN` parameters | 423,433 |
| `TaskA_CNN_LSTM` parameters | 949,769 |
| `FutureTimelineModel` parameters | 19,591 |
| `ToolPredictionModel` parameters | 17,799 |

Result files:

- `docs/results/project_summary.md`
- `docs/results/model_summary.csv`
- `docs/results/project_summary.json`
- `docs/results/structure_check.txt`

Model summary sample:

```csv
model,task,input_shape,output,parameters
TaskA_CNN,phase classification + remaining-time regression,"[batch, seq, 3, height, width]","phase logits, remaining-time ratio",423433
TaskA_CNN_LSTM,temporal phase classification + remaining-time regression,"[batch, seq, 3, height, width]","phase logits, remaining-time ratio",949769
```

![Remaining-time comparison](picture/compare.jpg)

## Quick Start

Set up the environment and run the lightweight structure check:

```bash
bash scripts/setup_env.sh
bash scripts/check_project.sh
```

Reuse an existing conda environment:

```bash
conda run -n codex_python bash scripts/check_project.sh
```

Generate the README result files:

```bash
make results
```

Training example:

```bash
python train_backbone.py --name backbone_cnn --epochs 25 --model cnn --data_root data/cholec80
python train_taskA_out_head.py --name timeline_head --epochs 20 --data_root data/cholec80
python train_taskB_out_head.py --name tool_head --epochs 20 --data_root data/cholec80
```

Workflow-aware weighting is enabled by default. To fall back to flat objectives:

```bash
python train_backbone.py --name flat_cnn --epochs 25 --model cnn --data_root data/cholec80 --disable_class_balance --phase_group_loss_weight 0 --phase_order_loss_weight 0
python train_taskA_out_head.py --name flat_timeline --epochs 20 --data_root data/cholec80 --timeline_loss_weighting uniform
python train_taskB_out_head.py --name flat_tool --epochs 20 --data_root data/cholec80 --disable_tool_class_balance
```

## Requirements

- Python 3.10+
- PyTorch
- Dependencies listed in `requirements.txt`

## Data Notes

- The Cholec80 dataset is not included in this repository.
- Official Cholec80 access point: [CAMMA-public/TF-Cholec80](https://github.com/CAMMA-public/TF-Cholec80); its preparation script downloads `https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz`.
- The default data path is `data/cholec80`.
- The official notes recommend around 166 GB of free space before download; the extracted dataset is about 85.2 GB.
- Full training and evaluation require `data/cholec80/frames/`, `data/cholec80/phase_annotations/`, and `data/cholec80/tool_annotations/`.
- Phase labels include Preparation, CalotTriangleDissection, ClippingCutting, GallbladderDissection, GallbladderPackaging, CleaningCoagulation, and GallbladderRetraction.
- Tool labels include Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, and SpecimenBag.
- `picture/compare.jpg` is a recorded experiment figure; with local data and checkpoints, regenerate it through `general_compare_diagram.py`.

## Data References

- [CAMMA-public/TF-Cholec80](https://github.com/CAMMA-public/TF-Cholec80): official Cholec80 data-preparation entry point, including the download script, dataset size, phase labels, and tool labels.
- [Twinanda et al., EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos](https://doi.org/10.1109/TMI.2016.2593957): Cholec80 dataset paper; the official data notes ask users to cite this paper when using Cholec80.

## Project Layout

```text
model_backbone.py          CNN and CNN-LSTM backbones
model_out_head.py          Timeline and tool output heads
workflow_schema.py         Phase, phase-group, and tool label definitions
workflow_losses.py         Hierarchical loss and weighting utilities
taskA_data_loader.py       Phase/time data loader
taskB_data_loader.py       Phase/time/tool data loader
train_*.py                 Training scripts
test_*.py                  Model evaluation scripts
picture/                   Method and result figures
docs/results/              Reproducible result summaries
tests/                     Lightweight tests
scripts/                   Setup, check, and result-generation scripts
```

## Tests

```bash
pytest tests/ -q
make test
```
