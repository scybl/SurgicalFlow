from pathlib import Path
import xml.etree.ElementTree as ET


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


def test_method_visual_assets_exist():
    preview = ROOT / "docs" / "images" / "surgical-flow-preview.svg"
    assert preview.is_file()
    ET.parse(preview)
    assert (ROOT / "picture" / "train_pipeline.png").is_file()
    assert (ROOT / "picture" / "compare.jpg").is_file()
