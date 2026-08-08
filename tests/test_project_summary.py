import csv
import json
from pathlib import Path

from scripts.generate_project_summary import model_specs, project_summary


ROOT = Path(__file__).resolve().parents[1]


def test_generated_model_summary_matches_script_values():
    expected = {spec.name: spec.parameters for spec in model_specs()}
    with (ROOT / "docs" / "results" / "model_summary.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    assert {row["model"]: int(row["parameters"]) for row in rows} == expected


def test_generated_project_summary_matches_script_values():
    expected = project_summary()
    actual = json.loads(
        (ROOT / "docs" / "results" / "project_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert actual["phase_classes"] == expected["phase_classes"]
    assert actual["tool_labels"] == expected["tool_labels"]
    assert actual["default_sequence_length"] == expected["default_sequence_length"]
    assert actual["default_stride"] == expected["default_stride"]
