from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_showcase_sections_and_english_companion():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_en.md").read_text(encoding="utf-8")

    assert "## 成果速览" in readme
    assert "## 简历亮点" in readme
    assert "## 复现边界" in readme
    assert "## Result Showcase" in english
    assert "## Resume Highlights" in english
    assert "## Reproducibility Boundaries" in english


def test_method_visual_assets_exist():
    preview = ROOT / "docs" / "images" / "surgical-flow-preview.svg"
    assert preview.is_file()
    ET.parse(preview)
    assert (ROOT / "picture" / "train_pipeline.png").is_file()
    assert (ROOT / "picture" / "compare.jpg").is_file()
