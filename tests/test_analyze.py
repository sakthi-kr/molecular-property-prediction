"""Offline train/validation-only tests using fictional molecules and split artifacts."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from molprop.analyze import (
    AnalysisError, _metrics, _read_baselines, _subgroup_summary, analyze,
    grouped_bootstrap_rmse_difference, grouped_cv_ridge, load_analysis_config,
    load_verified_inputs, nearest_train_tanimoto, split_conformal_ridge,
)
from molprop.prepare import prepare_dataset
from molprop.audit import audit_dataset
from molprop.train import fit_baselines, load_prepared
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]

# Shared synthetic structures from the existing baseline unit tests; no real
# dataset or held-out test targets are required to run CI.
MOLECULES = [
    "CCO", "CCN", "CCC", "c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1ccoc1",
    "c1ccsc1", "C1CCNCC1", "C1CCOCC1", "c1ccc2ccccc2c1", "C1CCC2CCCCC2C1",
    "c1cn[nH]c1", "c1ncc[nH]1", "c1ccc2[nH]ccc2c1", "C1CCCC1",
    "c1ccc2occc2c1", "c1ccc2sccc2c1", "Fc1ccccc1", "c1ccnc(c1)F",
    "C1CCOC1", "c1ccc2ncccc2c1", "C1CCSC1", "O=C1CCCC1", "CCCl",
    "CCBr", "CC(=O)O", "CC(C)O", "O=C(O)c1ccccc1", "c1ccc(Cl)cc1",
]


@pytest.fixture(scope="module")
def ready(tmp_path_factory):
    root = tmp_path_factory.mktemp("molecule-analysis")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "Lipophilicity.csv"
    with raw.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["CMPD_CHEMBLID", "exp", "smiles"], lineterminator="\n")
        writer.writeheader()
        for idx, smiles in enumerate(MOLECULES):
            writer.writerow({"CMPD_CHEMBLID": f"FIX{idx:03d}", "exp": f"{(idx % 13)*.18-.1:.6f}",
                             "smiles": smiles})
    import hashlib
    (raw_dir / "source_manifest.json").write_text(json.dumps({
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "rows": len(MOLECULES)
    }))
    audit_dir, processed, baseline = root / "audit", root / "processed", root / "baselines"
    audit_dataset(raw, audit_dir)
    prepare_dataset(raw, audit_dir, processed, ROOT / "configs/preparation.json")
    fit_baselines(processed, baseline, ROOT / "configs/models.json")
    config = json.loads((ROOT / "configs/analysis.json").read_text())
    config["prediction_interval_coverage"] = 0.7  # finite quantile on very small toy calibration groups
    config["bootstrap_repetitions"] = 100
    config["max_outliers"] = 5
    analysis_cfg = root / "analysis.json"
    analysis_cfg.write_text(json.dumps(config), encoding="utf-8")
    return root, processed, baseline, analysis_cfg


def test_config_rejects_bad_grid(tmp_path):
    obj = json.loads((ROOT / "configs/analysis.json").read_text())
    obj["ridge_morgan_alpha_grid"] = [0, 1]
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(obj))
    with pytest.raises(AnalysisError, match="Invalid analysis config"):
        load_analysis_config(f)


def test_max_tanimoto_known_values():
    train = np.asarray([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.uint8)
    val = np.asarray([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    sim, neighbors = nearest_train_tanimoto(train, val, ["a", "b"])
    assert sim == pytest.approx([1, 1/3, 0])
    assert neighbors == ["a", "a", "a"]


def test_cv_folds_and_split_conformal_disjoint():
    x = csr_matrix(np.eye(30, dtype=float))
    y = np.arange(30) / 10
    groups = np.array([f"g{i//2}" for i in range(30)], dtype=object)
    cv = grouped_cv_ridge(x, y, groups, [0.5, 1.0], 3)
    assert cv["selected_alpha"] in [0.5, 1.0]
    out, (prediction, low, high), _ = split_conformal_ridge(
        x, y, groups, x[:4], y[:4], alpha=1.0, fraction=0.4,
        coverage=0.75, seed=7,
    )
    assert out["training_groups"] + out["calibration_groups"] == len(set(groups))
    assert len(prediction) == 4 and (high >= low).all()


def test_paired_group_bootstrap_is_deterministic():
    y = np.asarray([0., 0., 1., 1.])
    first = np.asarray([0., 1., 1., 2.])
    second = np.asarray([0., 0., 0., 1.])
    kwargs = dict(n_boot=200, seed=2026)
    a = grouped_bootstrap_rmse_difference(y, first, second, ["a", "a", "b", "b"], **kwargs)
    b = grouped_bootstrap_rmse_difference(y, first, second, ["a", "a", "b", "b"], **kwargs)
    assert a == b and a["groups"] == 2
    assert a["approx_95pct_interval"][0] <= a["approx_95pct_interval"][1]


def test_analysis_writes_validation_only_and_preserves_inputs(ready, tmp_path):
    root, processed, baseline, cfg = ready
    before = {p: p.read_bytes() for p in [processed / "features.npz",
        processed / "scaffold_held_out.csv", baseline / "training_report.json"]}
    output = tmp_path / "analysis"
    result = analyze(processed, baseline, output, ROOT / "configs/models.json", cfg)
    assert result["test_evaluated"] is False
    assert result["test_predictions_written"] is False
    assert set(result["strategies"]) == {"random_grouped", "scaffold_held_out"}
    _, _, assignments = load_prepared(processed)
    for strategy, detail in result["strategies"].items():
        file = output / strategy / "validation_diagnostics.csv"
        with file.open(encoding="utf-8", newline="") as fh:
            items = list(csv.DictReader(fh))
        val_ids = {r["csv_row"] for r in assignments[strategy] if r["split"] == "validation"}
        test_ids = {r["csv_row"] for r in assignments[strategy] if r["split"] == "test"}
        assert {r["csv_row"] for r in items} == val_ids
        assert not test_ids.intersection({r["csv_row"] for r in items})
        assert detail["validation_rows"] == len(items)
        assert detail["prediction_interval"]["training_groups"] > 0
        assert detail["prediction_interval"]["calibration_groups"] > 0
        assert (output / strategy / "similarity_vs_error.png").exists()
        assert (output / strategy / "observed_vs_predicted.png").exists()
        assert (output / strategy / "cv_selected_ridge_morgan.joblib").exists()
    for p, original in before.items():
        assert p.read_bytes() == original
    with pytest.raises(AnalysisError, match="already exist"):
        analyze(processed, baseline, output, ROOT / "configs/models.json", cfg)


def test_baseline_prediction_tamper_detected(ready, tmp_path):
    root, processed, baseline, cfg = ready
    # Copy baseline report/predictions only, not model binaries; analysis must
    # validate saved CSVs and never deserialize joblib files.
    copy = tmp_path / "baseline"
    for strategy in ["random_grouped", "scaffold_held_out"]:
        (copy / strategy).mkdir(parents=True)
        shutil.copyfile(baseline / strategy / "validation_predictions.csv", copy / strategy / "validation_predictions.csv")
    shutil.copyfile(baseline / "training_report.json", copy / "training_report.json")
    path = copy / "random_grouped" / "validation_predictions.csv"
    text = path.read_text()
    path.write_text(text.replace("dummy_mean", "UNKNOWN_MODEL", 1))
    with pytest.raises(AnalysisError, match="Prediction model or compound ID"):
        analyze(processed, copy, tmp_path / "out", ROOT / "configs/models.json", cfg)


def test_input_config_tamper_detected(ready, tmp_path):
    _, processed, baseline, cfg = ready
    models_cfg = json.loads((ROOT / "configs/models.json").read_text())
    models_cfg["ridge_alpha_morgan"] = 12345
    bad = tmp_path / "models.json"
    bad.write_text(json.dumps(models_cfg))
    with pytest.raises(AnalysisError, match="model config"):
        analyze(processed, baseline, tmp_path / "out", bad, cfg)


def test_quartile_bins_and_empty_bins():
    observed = np.array([0., 1., 2.])
    predicted = np.array([0., 2., 1.])
    out = _subgroup_summary(observed, observed, predicted, [-np.inf, 0.5, 1.5, 3.], ["a", "b", "c"])
    assert [x["n"] for x in out] == [1, 1, 1]
    assert out[1]["mae"] == 1
