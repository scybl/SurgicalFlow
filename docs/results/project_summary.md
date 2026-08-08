# SurgicalFlow Project Summary

| Item | Value |
| --- | --- |
| Phase classes | 7 |
| Tool labels | 7 |
| Default sequence length | 16 frames |
| Default stride | 8 frames |

## Model Inventory

| Model | Task | Parameters |
| --- | --- | ---: |
| TaskA_CNN | phase classification + remaining-time regression | 423,433 |
| TaskA_CNN_LSTM | temporal phase classification + remaining-time regression | 949,769 |
| FutureTimelineModel | future phase boundary regression | 19,591 |
| ToolPredictionModel | multi-label tool presence prediction | 17,799 |

## Entrypoints

| Type | Scripts |
| --- | --- |
| Training | `train_backbone.py`, `train_taskA_out_head.py`, `train_taskB_out_head.py` |
| Evaluation | `test_backbone.py`, `test_taskA_out_head.py`, `test_taskB_out_head.py`, `general_compare_diagram.py`, `checkdata.py` |
