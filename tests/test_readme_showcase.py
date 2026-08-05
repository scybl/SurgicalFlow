from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_showcase_sections_and_english_companion():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_en.md").read_text(encoding="utf-8")

    assert "## 简历亮点" in readme
    assert "## 复现边界" in readme
    assert "## Resume Highlights" in english
    assert "## Reproducibility Boundaries" in english


def test_method_visual_assets_exist():
    assert (ROOT / "picture" / "train_pipeline.png").is_file()
    assert (ROOT / "picture" / "compare.jpg").is_file()
