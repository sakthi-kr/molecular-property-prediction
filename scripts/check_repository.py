"""Verify public repository hygiene without loading molecular data or models."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "LICENSE", "pyproject.toml", ".github/workflows/tests.yml",
    "docs/results.md", "docs/reproducibility.md", "docs/github_setup.md", "configs/final_evaluation.json",
    "src/molprop/predict.py", "src/molprop/app.py", "tests/test_predict.py",
    "examples/sample_smiles.csv",
)
PRIVATE_PREFIXES = ("outputs/", "models/", "data/raw/", "data/processed/", "data/reports/", ".venv/")
PRIVATE_SUFFIXES = (".joblib", ".npz", ".pkl", ".pickle", ".parquet")


def check() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required repository files: {', '.join(missing)}")
    # Outside git, the staged file list does not exist yet. After git add,
    # this checks the actual index (what will be committed), not ignored local files.
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "-z"], cwd=ROOT,
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        print("PASS: required source files present; git-index privacy check requires git init/add")
        return
    tracked = [part.decode("utf-8", errors="replace").replace("\\", "/")
               for part in proc.stdout.split(b"\0") if part]
    leaks = [path for path in tracked if (
        (path.startswith(PRIVATE_PREFIXES) and not path.endswith("/.gitkeep"))
        or path.endswith(PRIVATE_SUFFIXES)
        or (path.endswith(".csv") and path != "examples/sample_smiles.csv")
    )]
    if leaks:
        raise RuntimeError("Refusing generated/data files in Git index: " + ", ".join(leaks))
    print(f"PASS: source layout and tracked-file hygiene verified ({len(tracked)} Git entries)")


if __name__ == "__main__":
    try:
        check()
    except (RuntimeError, OSError) as exc:
        sys.exit(f"FAIL: {exc}")
