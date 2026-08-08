from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MARKERS = (
    "mp" + "hy" + "00" + "43",
    "u" + "cl/",
    "u" + "cl ",
    "university" + " college london",
    "course" + "work",
    "assign" + "ment",
    "moo" + "dle",
    "简历" + "亮点",
)
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
SKIPPED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def iter_project_text_files():
    for path in ROOT.rglob("*"):
        if any(part in SKIPPED_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def test_project_text_has_no_public_repo_markers():
    offenders = []
    for path in iter_project_text_files():
        text = path.read_text(encoding="utf-8").lower()
        relative_path = path.relative_to(ROOT).as_posix().lower()
        if any(marker in text or marker in relative_path for marker in FORBIDDEN_MARKERS):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
