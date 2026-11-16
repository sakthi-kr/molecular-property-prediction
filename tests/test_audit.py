"""Source quality tests with tiny fictional molecular inputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from molprop.audit import DataAuditError, audit_dataset


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["CMPD_CHEMBLID", "exp", "smiles"])
        writer.writeheader()
        writer.writerows(rows)


def test_audit_flags_canonical_duplicates_and_conflicting_targets(tmp_path: Path):
    source = tmp_path / "raw.csv"
    rows = [
        {"CMPD_CHEMBLID": "TEST1", "exp": "1.1", "smiles": "CCO"},
        {"CMPD_CHEMBLID": "TEST2", "exp": "1.9", "smiles": "OCC"},
        {"CMPD_CHEMBLID": "TEST3", "exp": "1.2", "smiles": "C1CCCCC1"},
        {"CMPD_CHEMBLID": "TEST4", "exp": "", "smiles": "CC"},
        {"CMPD_CHEMBLID": "TEST5", "exp": "2.1", "smiles": "bad-random"},
        {"CMPD_CHEMBLID": "TEST6", "exp": "nan", "smiles": "CCO"},
    ]
    write_csv(source, rows)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = audit_dataset(source, tmp_path / "reports")
    assert result["raw_rows"] == 6
    assert result["valid_rows"] == 3
    assert result["invalid_rows"] == 3
    assert result["unique_canonical_smiles"] == 2
    assert result["duplicate_canonical_groups"] == 1
    assert result["conflicting_label_groups"] == 1
    assert result["source_sha256"] == digest
    assert result["distinct_scaffolds"] == 2
    assert result["acyclic_rows"] == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest  # source unchanged
    saved = json.loads((tmp_path / "reports/data_audit.json").read_text())
    assert saved["duplicate_canonical_groups"] == 1
    assert (tmp_path / "reports/valid_records.csv").is_file()
    assert (tmp_path / "reports/invalid_records.csv").is_file()


def test_audit_rejects_missing_column(tmp_path: Path):
    path = tmp_path / "broken.csv"
    path.write_text("smiles,target\nCCO,1.2\n")
    with pytest.raises(DataAuditError, match="Expected columns"):
        audit_dataset(path, tmp_path / "out")


def test_audit_rejects_missing_dataset(tmp_path: Path):
    with pytest.raises(DataAuditError, match="not found"):
        audit_dataset(tmp_path / "absent.csv", tmp_path / "reports")


def test_audit_rejects_empty_dataset(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("smiles,exp\n")
    with pytest.raises(DataAuditError, match="no rows"):
        audit_dataset(path, tmp_path / "reports")


def test_audit_handles_all_invalid_rows_without_division_error(tmp_path: Path):
    path = tmp_path / "invalid.csv"
    path.write_text("smiles,exp\n,1.2\nCCO,NA\n")
    r = audit_dataset(path, tmp_path / "r")
    assert r["valid_rows"] == 0
    assert r["target_summary"]["mean"] is None
    assert r["invalid_rows"] == 2


def test_audit_keeps_identical_labels_as_duplicate_not_conflict(tmp_path: Path):
    path = tmp_path / "dups.csv"
    path.write_text("smiles,exp\nCCO,0.25\nOCC,0.25\n")
    r = audit_dataset(path, tmp_path / "r")
    assert r["duplicate_canonical_groups"] == 1
    assert r["conflicting_label_groups"] == 0
