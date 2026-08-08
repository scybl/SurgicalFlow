from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_showcase_sections_and_english_companion():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_en.md").read_text(encoding="utf-8")

    assert "## 功能说明" in readme
    assert "## 结果展示" in readme
    assert "## 快速上手" in readme
    assert "## 环境要求" in readme
    assert "## 数据说明" in readme
    assert "## Results" in english
    assert "## Features" in english
    assert "## Quick Start" in english
    assert "## Requirements" in english
    assert "## Data Notes" in english


def test_readme_uses_reproducible_results_and_real_assets():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_en.md").read_text(encoding="utf-8")
    decorative_preview = f"docs/images/{'surgical-flow'}-preview.svg"

    assert decorative_preview not in readme
    assert decorative_preview not in english
    assert "docs/results/model_summary.csv" in readme
    assert "docs/results/model_summary.csv" in english

    assert (ROOT / "picture" / "train_pipeline.png").is_file()
    assert (ROOT / "picture" / "compare.jpg").is_file()
    assert (ROOT / "docs" / "results" / "project_summary.md").is_file()
    assert (ROOT / "docs" / "results" / "model_summary.csv").is_file()


def test_readme_has_cholec80_download_notes():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_en.md").read_text(encoding="utf-8")

    assert "https://github.com/CAMMA-public/TF-Cholec80" in readme
    assert "https://github.com/CAMMA-public/TF-Cholec80" in english
    assert "cholec80.tar.gz" in readme
    assert "cholec80.tar.gz" in english
    assert "data/cholec80/frames/" in readme
    assert "data/cholec80/frames/" in english
