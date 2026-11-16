"""Offline tests of model baselines with toy fictional molecules and locked splits."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from molprop.audit import audit_dataset
from molprop.prepare import prepare_dataset
from molprop.train import TrainingError, fit_baselines, load_prepared, _metrics

ROOT = Path(__file__).resolve().parents[1]
PREP_CONFIG = ROOT / "configs/preparation.json"
MODELS_CONFIG = ROOT / "configs/models.json"

MOLECULES = [
    "CCO", "CCN", "CCC", "c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1ccoc1",
    "c1ccsc1", "C1CCNCC1", "C1CCOCC1", "c1ccc2ccccc2c1", "C1CCC2CCCCC2C1",
    "c1cn[nH]c1", "c1ncc[nH]1", "c1ccc2[nH]ccc2c1", "C1CCCC1",
    "c1ccc2occc2c1", "c1ccc2sccc2c1", "Fc1ccccc1", "c1ccnc(c1)F",
    "C1CCOC1", "c1ccc2ncccc2c1", "C1CCSC1", "O=C1CCCC1", "CCCl",
    "CCBr", "CC(=O)O", "CC(C)O", "O=C(O)c1ccccc1", "c1ccc(Cl)cc1",
]


def prepared(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "Lipophilicity.csv"
    with raw.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["CMPD_CHEMBLID", "exp", "smiles"], lineterminator="\n")
        writer.writeheader()
        for i, smiles in enumerate(MOLECULES):
            writer.writerow({"CMPD_CHEMBLID":f"FIX{i:03d}","exp":f"{(i % 13)*0.18-0.1:.5f}", "smiles":smiles})
    (raw_dir / "source_manifest.json").write_text(json.dumps({
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "rows":len(MOLECULES)
    }))
    audit_dir = tmp_path / "report"
    audit_dataset(raw, audit_dir)
    processed_dir = tmp_path / "processed"
    prepare_dataset(raw, audit_dir, processed_dir, PREP_CONFIG)
    return processed_dir


def test_train_baselines_never_write_test_predictions(tmp_path: Path):
    folder = prepared(tmp_path)
    target = tmp_path / "outputs"
    result = fit_baselines(folder, target, MODELS_CONFIG)
    assert result["test_evaluated"] is False
    assert result["test_predictions_written"] is False
    assert set(result["models"]) == {"random_grouped", "scaffold_held_out"}
    for strategy in result["models"]:
        entries = result["models"][strategy]["candidates_by_validation_rmse"]
        assert len(entries) == 4
        assert all(x["train"]["rmse"] >= 0 and x["validation"]["rmse"] >= 0 for x in entries)
        assert result["models"][strategy]["best_validation_model"] == entries[0]["model"]
        assert (target / strategy / "ridge_descriptors.joblib").exists()
        with (target / strategy / "validation_predictions.csv").open(newline="",encoding="utf-8") as fp:
            preds = list(csv.DictReader(fp))
        with (folder / f"{strategy}.csv").open(newline="", encoding="utf-8") as fp:
            spl = list(csv.DictReader(fp))
        ids = {r["csv_row"] for r in spl if r["split"] == "validation"}
        test_ids = {r["csv_row"] for r in spl if r["split"] == "test"}
        assert {r["csv_row"] for r in preds} == ids
        assert {r["csv_row"] for r in preds}.isdisjoint(test_ids)
        assert len(preds) == 4*len(ids)
    assert (target / "training_report.json").exists()


def test_repeat_requires_explicit_force(tmp_path: Path):
    processed = prepared(tmp_path)
    out = tmp_path / "out"
    a=fit_baselines(processed,out,MODELS_CONFIG)
    with pytest.raises(TrainingError,match="already exist"):
        fit_baselines(processed,out,MODELS_CONFIG)
    b=fit_baselines(processed,out,MODELS_CONFIG,force=True)
    for strategy in a["models"]:
        assert a["models"][strategy]["candidates_by_validation_rmse"] == b["models"][strategy]["candidates_by_validation_rmse"]


def test_modified_features_detected_by_checksum(tmp_path: Path):
    p=prepared(tmp_path)
    with (p / "features.npz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(TrainingError,match="Checksum mismatch"):
        load_prepared(p)


def test_modified_split_detected_by_checksum(tmp_path: Path):
    p=prepared(tmp_path)
    with (p / "random_grouped.csv").open("a") as f:
        f.write("extra row\n")
    with pytest.raises(TrainingError,match="Checksum mismatch"):
        load_prepared(p)


def test_models_config_validation(tmp_path: Path):
    p=prepared(tmp_path)
    model_cfg = json.loads(MODELS_CONFIG.read_text())
    model_cfg["ridge_alpha_morgan"] = -1
    path=tmp_path/"bad.json"
    path.write_text(json.dumps(model_cfg))
    with pytest.raises(TrainingError,match="Invalid fixed model configuration"):
        fit_baselines(p,tmp_path/"out",path)


def test_metrics_exact_on_simple_arrays():
    result=_metrics(np.array([1.,2.,3.]),np.array([1.,2.,3.]))
    assert result == {"rmse":0.0,"mae":0.0,"r2":1.0}
