"""Public checkout guardrails; no source dataset or model artifacts required."""
from __future__ import annotations

import csv
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]


def test_example_molecules_valid():
    with (ROOT / "examples/sample_smiles.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 3
    assert all(x.get("compound_id") and Chem.MolFromSmiles(x["smiles"]) for x in records)


def test_release_documentation_exists():
    for path in ["docs/results.md", "docs/reproducibility.md", "README.md", "LICENSE"]:
        assert (ROOT / path).is_file()


def test_generated_data_excluded_from_version_control():
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for excluded in ["data/raw/*", "data/processed/*", "outputs/", "*.joblib", "*.npz", "*.csv"]:
        assert excluded in rules
