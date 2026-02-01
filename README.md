# Analysis of the Impact of Temporal Window Length on Multi-task Surgical Workflow Prediction

This work analyzes the impact of temporal window length on multi-task surgical workflow prediction.

## Abstract

This work use to jointly perform future phase boundary estimation (Task A) and currentphase tool usage recognition (Task B) on the Cholec80 dataset. Experimental results show that moderate temporal windows improve the stability of future phase boundary prediction, while excessively long windows do not yield additional benefits and may degrade phase recognition performance. Tool usage prediction remains stable across temporal settings, achieving 246.02s MAE and 0.62 R².

## Overall Project Structure

The following is the layout table of the project

```bash
.
├── README.md
├── checkdata.py										# Retrieve the status distribution of the dataset
├── data														# Location of the dataset
│   └── cholec80
│       ├── frames
│       │   ├── video01
│       │   ···
│       ├── phase_annotations
│       │   ├── video01-phase.txt
│       │   ···
│       └── tool_annotations
│           ├── video01-tool.txt
│           ···
├── model_backbone.py							 # Model backbone file
├── model_out_head.py							 # Task A and Task B output head
├── general_compare_diagram.py		 # general the result diagram										
├── taskB_data_loader.py					 # Task A of data loader 
├── taskB_data_loader.py					 # Task B of data loader 
├── test_backbone.py							 # test the performance of backbone
├── test_taskA_out_head.py				 # test the performance of Task A output head
├── test_taskB_out_head.py				 # test the performance of Task B output head
├── train_backbone.py							 # train backbone model
├── train_taskA_out_head.py				 # train Task A output head
└── train_taskB_out_head.py				 # train Task A output head
```

## Dataset Prepare

When preparing the dataset, it is recommended to use the official github download method. The following is the method for preparing the dataset

```bash
git clone git@github.com:CAMMA-public/TF-Cholec80.git
python prepare.py --data_rootdir <YOUR_LOCATION>
```

Then move the dataset to the '<project_root>/data' directory of the project

## Environment Setup

When configuring the environment, the following code can be used for environment configuration. It is recommended to use micromamba or conda

### Option 1: Micromamba (Recommended)

```
micromamba create -n cholec80_env python=3.10
micromamba activate cholec80_env

pip install torch torchvision numpy scikit-learn Pillow matplotlib tqdm
```

---

### Option 2: Conda (Lightweight Alternative)

```
conda create -n cholec80_env python=3.10
conda activate cholec80_env

pip install torch torchvision numpy scikit-learn Pillow matplotlib tqdm
```

------

The environment can be configured according to the above methods

## Train Model

This work dynamically controls the training method by using the 'argparse' approach

```bash
python train_backbone.py --name backbone_cnn --epochs 25 --model cnn # train cnn backbone model
python train_backbone.py --name backbone_cnn_lstm_16 --epochs 25 --model cnn_lstm # train cnn backbone model
python train_backbone.py --name backbone_cnn_lstm_32 --epochs 25 --model cnn_lstm --seq_len 32 --stride 16 # train cnn backbone model 

python train_taskA_out_head.py --name taskA_out_head --epochs 25 # train taskA output head
python train_taskB_out_head.py --name taskB_out_head --epochs 25 # train taskA output head
```

In addition, there are other parameters available for selection in this training method

| Argument        | Description                                                  | Default         |
| --------------- | ------------------------------------------------------------ | --------------- |
| `--name`        | Experiment name. Used to identify the experiment and as the subdirectory name for saving logs and results | **Required**    |
| `--epochs`      | Number of training epochs                                    | **Required**    |
| `--model`       | **Only for train_backbone.py**, Backbone model type. Must be one of **`cnn` or `cnn_lstm`** | **Required**    |
| `--data_root`   | Root directory of the Cholec80 dataset                       | `data/cholec80` |
| `--batch_size`  | Batch size used during training                              | `16`            |
| `--lr`          | Initial learning rate (used by the Adam optimizer)           | `1e-4`          |
| `--seq_len`     | Temporal input window length (sequence length for LSTM / sliding window size) | `16`            |
| `--stride`      | Sliding window stride, controlling the overlap between consecutive samples | `8`             |
| `--num_workers` | Number of worker threads for parallel data loading           | `8`             |
| `--save_dir`    | Root directory for saving model checkpoints and training logs | `checkpoints`   |
| `--device`      | Training device. CUDA is used by default and automatically falls back to CPU if unavailable | `cuda`          |
| `--seed`        | Random seed for reproducibility                              | `42`            |

## Test Model

The following is the test code

```bash
python test_backbone.py --name backbone_cnn --model cnn # test cnn backbone model
python test_backbone.py --name backbone_cnn_lstm_16 --model cnn_lstm # test cnn_lstm_16 backbone model
python test_backbone.py --name backbone_cnn_lstm_32 --model cnn_lstm --seq_len 32 --stride 16 # test cnn_lstm_32 backbone model

python test_taskA_out_head.py --backbone_name backbone_cnn --backbone_model cnn --head_name taskA_out_head  # test taskA out head with cnn backbone
python test_taskA_out_head.py --backbone_name backbone_cnn_lstm_16 --backbone_model cnn_lstm --head_name taskA_out_head # test taskA out head with cnn_lstm_16 backbone
python test_taskA_out_head.py --backbone_name backbone_cnn_lstm_32 --backbone_model cnn_lstm --head_name taskA_out_head --seq_len 32 --stride 16 # test taskA out head with cnn_lstm_32 backbone

python test_taskB_out_head.py --backbone_name backbone_cnn --backbone_model cnn --head_name taskA_out_head # test taskB out head with cnn backbone
python test_taskA_out_head.py --backbone_name backbone_cnn_lstm_16 --backbone_model cnn_lstm --head_name taskA_out_head # test taskA out head with cnn_lstm_16 backbone
python test_taskA_out_head.py --backbone_name backbone_cnn_lstm_32 --backbone_model cnn_lstm --head_name taskA_out_head --seq_len 32 --stride 16 # test taskA out head with cnn_lstm_32 backbone
```

The following is the parameter table of the test code

| Argument           | Description                                                  | Default         |
| ------------------ | ------------------------------------------------------------ | --------------- |
| `--name`           | **Only for `test_backbone.py`**. Experiment name. Used to identify the experiment and as the subdirectory name for saving logs and results | **Required**    |
| `--model`          | **Only for `test_backbone.py`**, Backbone model type. Must be one of `cnn` or `cnn_lstm` | **Required**    |
| `--backbone_name`  | **Only for `test_taskA_out_head.py`** and **`test_taskB_out_head.py`**. Name of the backbone experiment. Used to identify the pretrained backbone checkpoint and its result directory | **Required**    |
| `--head_name`      | **Only for `test_taskA_out_head.py`** and** `test_taskB_out_head.py`**. Name of the head experiment. Used to identify the output head configuration and result directory | **Required**    |
| `--backbone_model` | Only for `test_taskA_out_head.py` and `test_taskB_out_head.py`, Backbone network type. Must be one of `cnn` or `cnn_lstm` | **Required**    |
| `--data_root`      | Root directory of the Cholec80 dataset                       | `data/cholec80` |
| `--seq_len`        | Temporal input window length (sequence length for temporal modeling) | `16`            |
| `--stride`         | Sliding window stride used for sequence sampling             | `8`             |
| `--batch_size`     | Batch size used during training                              | `16`            |
| `--lr`             | Initial learning rate (used by the Adam optimizer)           | `1e-4`          |
| `--epochs`         | Number of training epochs                                    | `20`            |
| `--num_workers`    | Number of worker threads for parallel data loading           | `8`             |
| `--device`         | Training device. CUDA is used by default and automatically falls back to CPU if unavailable | `cuda`          |
| `--seed`           | Random seed for reproducibility                              | `42`            |

## Known Issues and Notes

- A GPU with at least **8GB of memory** is recommended for training CNN-LSTM models to avoid out-of-memory errors.  
- Increasing the temporal window length significantly raises GPU memory usage and training time.  
- Video frames are uniformly sampled at **1 FPS** to reduce computational cost while preserving sufficient temporal information.  If you want other parameters, you can make targeted changes in dataloader.
- Extremely large temporal windows may introduce redundant temporal context and negatively affect model stability.

## Temporal Window Experiment Design

This project investigates the influence of temporal window length on multi-task surgical workflow prediction performance. To ensure fair comparison, all models are trained and evaluated under identical experimental settings, except for the temporal window configuration.

Three temporal modeling strategies are evaluated:

- **CNN Baseline**  
  A frame-based model without explicit temporal modeling. Each frame is processed independently and serves as a reference baseline.

- **CNN-LSTM (Window Length = 16)**  
  A temporal model that incorporates short-term temporal context by processing sequences of 16 consecutive frames.

- **CNN-LSTM (Window Length = 32)**  
  A temporal model that uses a longer temporal context by processing sequences of 32 consecutive frames.

For the CNN-LSTM models, sliding window sampling is applied with corresponding stride values to generate overlapping temporal sequences. All experiments use the same dataset split, training hyperparameters, and evaluation metrics to isolate the impact of temporal window length. The specific structure is shown in the following figure.

<img src="/Users/libingze/Desktop/周五早九/test_repo/picture/train_pipeline.png" alt="train_pipeline" style="zoom:100%;" />

This experimental design enables a systematic analysis of how temporal context length affects phase boundary prediction accuracy, remaining time estimation performance, and tool usage recognition stability.

## Experiments Result

This work systematically investigated the impact of temporal window length on multi-head surgical workflow prediction using a unified CNN-based and CNN-LSTM-based framework on the Cholec80 dataset. By jointly modeling surgical phase recognition, remaining time estimation, and tool usage prediction, providing a comprehensive evaluation of how temporal context influences different prediction objectives.

Experimental results show that a moderate temporal window (N = 16) achieves a favorable trade-off between modeling capability and prediction stability, improving remaining time regression performance while maintaining competitive accuracy for phase and tool recognition. In contrast, longer temporal windows introduce redundant temporal information and may degrade classification performance, highlighting the importance of selecting appropriate temporal input ranges for online surgical assistance systems.

<img src="picture/compare.jpg" alt="compare" style="zoom:60%;" />

These findings suggest that lightweight temporal modeling with constrained context length can achieve strong performance without unnecessary computational overhead. Future work will explore more advanced temporal architectures and adaptive window strategies.

