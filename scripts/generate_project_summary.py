#!/usr/bin/env python
"""Generate reproducible, data-free SurgicalFlow summary artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_schema import NUM_PHASES, NUM_TOOLS, PHASE_GROUPS

DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE = 8


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    input_shape: str
    output: str
    parameters: int


def linear_params(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def conv2d_params(in_channels: int, out_channels: int, kernel_size: int) -> int:
    return out_channels * in_channels * kernel_size * kernel_size + out_channels


def batchnorm2d_params(channels: int) -> int:
    return channels * 2


def lstm_params(input_size: int, hidden_size: int, layers: int = 1) -> int:
    total = 0
    for layer in range(layers):
        layer_input = input_size if layer == 0 else hidden_size
        total += 4 * hidden_size * layer_input
        total += 4 * hidden_size * hidden_size
        total += 8 * hidden_size
    return total


def cnn_feature_params() -> int:
    channels = ((3, 32), (32, 64), (64, 128), (128, 256))
    return sum(
        conv2d_params(in_channels, out_channels, 3)
        + batchnorm2d_params(out_channels)
        for in_channels, out_channels in channels
    )


def backbone_head_params(feature_dim: int) -> int:
    return (
        linear_params(feature_dim, 128)
        + linear_params(128, NUM_PHASES + 1)
        + linear_params(128, 1)
    )


def model_specs() -> list[ModelSpec]:
    cnn_features = cnn_feature_params()
    return [
        ModelSpec(
            name="TaskA_CNN",
            task="phase classification + remaining-time regression",
            input_shape="[batch, seq, 3, height, width]",
            output="phase logits, remaining-time ratio",
            parameters=cnn_features + backbone_head_params(256),
        ),
        ModelSpec(
            name="TaskA_CNN_LSTM",
            task="temporal phase classification + remaining-time regression",
            input_shape="[batch, seq, 3, height, width]",
            output="phase logits, remaining-time ratio",
            parameters=cnn_features + lstm_params(256, 256) + backbone_head_params(256),
        ),
        ModelSpec(
            name="FutureTimelineModel",
            task="future phase boundary regression",
            input_shape="[current phase, ratio, stage order, phase durations]",
            output="7 cumulative future timestamps",
            parameters=linear_params(16, 128)
            + linear_params(128, 128)
            + linear_params(128, NUM_PHASES),
        ),
        ModelSpec(
            name="ToolPredictionModel",
            task="multi-label tool presence prediction",
            input_shape="[current phase, remaining time]",
            output="7 tool logits",
            parameters=linear_params(2, 128)
            + linear_params(128, 128)
            + linear_params(128, NUM_TOOLS),
        ),
    ]


def project_summary() -> dict[str, object]:
    return {
        "project": "SurgicalFlow",
        "phase_classes": NUM_PHASES,
        "phase_groups": len(PHASE_GROUPS),
        "tool_labels": NUM_TOOLS,
        "default_sequence_length": DEFAULT_SEQ_LEN,
        "default_stride": DEFAULT_STRIDE,
        "workflow_aware_training": [
            "phase class balancing",
            "coarse phase-group loss",
            "ordinal phase-distance loss",
            "timeline horizon weighting",
            "tool positive-class weighting",
        ],
        "training_entrypoints": [
            "train_backbone.py",
            "train_taskA_out_head.py",
            "train_taskB_out_head.py",
        ],
        "evaluation_entrypoints": [
            "test_backbone.py",
            "test_taskA_out_head.py",
            "test_taskB_out_head.py",
            "general_compare_diagram.py",
            "checkdata.py",
        ],
    }


def write_model_csv(path: Path, specs: list[ModelSpec]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "task", "input_shape", "output", "parameters"],
            lineterminator="\n",
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "model": spec.name,
                    "task": spec.task,
                    "input_shape": spec.input_shape,
                    "output": spec.output,
                    "parameters": spec.parameters,
                }
            )


def write_markdown(path: Path, summary: dict[str, object], specs: list[ModelSpec]) -> None:
    lines = [
        "# SurgicalFlow Project Summary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Phase classes | {summary['phase_classes']} |",
        f"| Workflow phase groups | {summary['phase_groups']} |",
        f"| Tool labels | {summary['tool_labels']} |",
        f"| Default sequence length | {summary['default_sequence_length']} frames |",
        f"| Default stride | {summary['default_stride']} frames |",
        f"| Workflow-aware training | {', '.join(summary['workflow_aware_training'])} |",
        "",
        "## Model Inventory",
        "",
        "| Model | Task | Parameters |",
        "| --- | --- | ---: |",
    ]
    for spec in specs:
        lines.append(f"| {spec.name} | {spec.task} | {spec.parameters:,} |")

    training = ", ".join(f"`{item}`" for item in summary["training_entrypoints"])
    evaluation = ", ".join(f"`{item}`" for item in summary["evaluation_entrypoints"])
    lines.extend(
        [
            "",
            "## Entrypoints",
            "",
            "| Type | Scripts |",
            "| --- | --- |",
            f"| Training | {training} |",
            f"| Evaluation | {evaluation} |",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = project_summary()
    specs = model_specs()

    write_model_csv(args.output_dir / "model_summary.csv", specs)
    write_markdown(args.output_dir / "project_summary.md", summary, specs)
    (args.output_dir / "project_summary.json").write_text(
        json.dumps(
            {
                **summary,
                "models": [spec.__dict__ for spec in specs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote summary artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
