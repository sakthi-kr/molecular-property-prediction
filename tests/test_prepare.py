"""Offline tests for feature arrays, source provenance and held-out splits."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from molprop.audit import audit_dataset
from molprop.prepare import PreparationError, prepare_dataset
from molprop.representation import DESCRIPTOR_NAMES, calculate_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preparation.json"

# Distinct scaffolds, plus one pair of canonical duplicates with identical target.
MOLECULES = [
    "CCO", "OCC", "c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1ccoc1",
    "c1ccsc1", "C1CCNCC1", "C1CCOCC1", "c1ccc2ccccc2c1",
    "C1CCC2CCCCC2C1", "c1cn[nH]c1", "c1ncc[nH]1", "c1ccc2[nH]ccc2c1",
    "C1CCCC1", "c1ccc2occc2c1", "c1ccc2sccc2c1", "Fc1ccccc1",
    "c1ccnc(c1)F", "C1CCOC1", "c1ccc2ncccc2c1", "C1CCSC1",
]


def fixture(tmp_path: Path, *, duplicates=True):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_csv = raw_dir / "Lipophilicity.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["CMPD_CHEMBLID", "exp", "smiles"], lineterminator="\n")
        w.writeheader()
        for i, smi in enumerate(MOLECULES if duplicates else MOLECULES[2:]):
            w.writerow({"CMPD_CHEMBLID":f"TEST{i+1}", "exp": "1.0" if i < 2 else f"{1+i/20:.4f}", "smiles":smi})
    manifest = {"sha256": hashlib.sha256(raw_csv.read_bytes()).hexdigest(), "rows":len(MOLECULES)}
    (raw_dir / "source_manifest.json").write_text(json.dumps(manifest))
    audit_dir = tmp_path / "reports"
    audit_dataset(raw_csv, audit_dir)
    return raw_csv, audit_dir


def test_feature_types_determinism_and_molecular_variation():
    xs = ["CCO", "C1CCCCC1", "c1ccccc1"]
    a,b = calculate_features(xs,radius=2,bits=256,include_chirality=True)
    a2,b2 = calculate_features(xs,radius=2,bits=256,include_chirality=True)
    assert a.shape == (3,len(DESCRIPTOR_NAMES))
    assert b.shape == (3,256)
    assert a.dtype == np.float64
    assert b.dtype == np.uint8
    assert set(np.unique(b)).issubset({0,1})
    np.testing.assert_array_equal(a,a2)
    np.testing.assert_array_equal(b,b2)
    assert not np.array_equal(b[0], b[1])


def test_prepare_locked_splits_and_no_raw_modification(tmp_path):
    raw, audit_dir = fixture(tmp_path)
    before_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    out = tmp_path / "processed"
    report = prepare_dataset(raw,audit_dir,out,CONFIG)
    assert report["rows"] == len(MOLECULES)
    assert report["descriptor_count"] == len(DESCRIPTOR_NAMES)
    assert report["splits"]["random_grouped"]["canonical_groups_crossing_partitions"] == 0
    assert report["splits"]["scaffold_held_out"]["scaffolds_crossing_partitions"] == 0
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before_hash
    assert (out / "preparation_report.json").exists()
    with np.load(out / "features.npz") as arrays:
        assert arrays["descriptors"].shape == (len(MOLECULES), len(DESCRIPTOR_NAMES))
        assert arrays["morgan"].shape == (len(MOLECULES),2048)
        assert np.isfinite(arrays["descriptors"]).all()
        assert np.isfinite(arrays["logd"]).all()
        assert arrays["csv_row"].tolist() == list(range(2,2+len(MOLECULES)))
    for name in ("random_grouped", "scaffold_held_out"):
        with (out/f"{name}.csv").open(newline="",encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows)==len(MOLECULES)
        assert set(r["split"] for r in rows)=={"train","validation","test"}
    with (out/"random_grouped.csv").open(newline="",encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["split"] == rows[1]["split"]  # CCO, OCC are the same molecule


def test_repeat_requires_force_and_is_deterministic(tmp_path):
    raw,audit_dir=fixture(tmp_path)
    output=tmp_path/"processed"
    r1=prepare_dataset(raw,audit_dir,output,CONFIG)
    a=(output/"random_grouped.csv").read_bytes()
    b=(output/"scaffold_held_out.csv").read_bytes()
    with pytest.raises(PreparationError,match="already exist"):
        prepare_dataset(raw,audit_dir,output,CONFIG)
    r2=prepare_dataset(raw,audit_dir,output,CONFIG,force=True)
    assert r1["splits"] == r2["splits"]
    assert a==(output/"random_grouped.csv").read_bytes()
    assert b==(output/"scaffold_held_out.csv").read_bytes()


def test_source_mutation_invalidates_provenance(tmp_path):
    raw,audit_dir=fixture(tmp_path)
    raw.write_text(raw.read_text()+"\n")
    with pytest.raises(PreparationError,match="changed since download"):
        prepare_dataset(raw,audit_dir,tmp_path/"out",CONFIG)


def test_conflicting_duplicates_block_modelling(tmp_path):
    raw,audit_dir=fixture(tmp_path)
    lines=raw.read_text().splitlines()
    lines[2]=lines[2].replace(",1.0,",",3.4,")
    raw.write_text("\n".join(lines)+"\n")
    manifest=raw.with_name("source_manifest.json")
    manifest.write_text(json.dumps({"sha256":hashlib.sha256(raw.read_bytes()).hexdigest()}))
    audit_dataset(raw,audit_dir)
    with pytest.raises(PreparationError,match="Conflicting measurements"):
        prepare_dataset(raw,audit_dir,tmp_path/"out",CONFIG)


def test_missing_preflight_inputs_rejected(tmp_path):
    with pytest.raises(PreparationError,match="Required local input"):
        prepare_dataset(tmp_path/"raw.csv",tmp_path/"reports",tmp_path/"processed",CONFIG)
