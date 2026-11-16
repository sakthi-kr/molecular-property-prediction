"""Test the frozen protocol on fictional molecules, never on reserved real test labels."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn import __version__ as sklearn_version

from molprop.analyze import analyze
from molprop.final_evaluation import (
    FinalEvaluationError, final_evaluation, load_lock,
)
from molprop.prepare import digest
from molprop.train import SPLIT_STRATEGIES, load_prepared

# Reuse the *fictional* dataset and fitted models from the preceding unit suite.
from test_analyze import ready  # noqa: F401 -- pytest fixture imported explicitly

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def locked_toy(ready, tmp_path_factory):
    root, processed, baseline, config = ready
    directory = tmp_path_factory.mktemp("locked-final-test")
    analysis_dir = directory / "analysis"
    analyzed = analyze(processed, baseline, analysis_dir, ROOT / "configs/models.json", config)
    prep = json.loads((processed / "preparation_report.json").read_text())
    model_cfg = json.loads((ROOT / "configs/models.json").read_text())
    lock = json.loads((ROOT / "configs/final_evaluation.json").read_text())
    lock.update({
        "dataset": prep["dataset"],
        "preparation_report_sha256": digest(processed / "preparation_report.json"),
        "training_report_sha256": digest(baseline / "training_report.json"),
        "analysis_report_sha256": digest(analysis_dir / "analysis_report.json"),
        "prepared_features_sha256": prep["data_sha256"]["features.npz"],
        "model_config_sha256": digest(ROOT / "configs/models.json"),
        "analysis_config_sha256": digest(config),
        "expected_sklearn_version": sklearn_version,
        "expected_test_rows_per_split": prep["splits"]["scaffold_held_out"]["rows"]["test"],
        "ridge_alpha": model_cfg["ridge_alpha_morgan"],
        "nominal_interval_coverage": analyzed["strategies"]["scaffold_held_out"]["prediction_interval"]["nominal_coverage"],
        "cv_selected_alpha_by_split": {
            s: analyzed["strategies"][s]["cv_training_only"]["selected_alpha"]
            for s in SPLIT_STRATEGIES
        },
    })
    lock_file = directory / "final_evaluation.json"
    lock_file.write_text(json.dumps(lock, indent=2) + "\n")
    return processed, baseline, analysis_dir, config, lock_file


def run(toy, target, **kwargs):
    processed, baseline, analysis_dir, analysis_config, lock_file = toy
    return final_evaluation(
        processed, baseline, analysis_dir, target, lock_file,
        ROOT / "configs/models.json", analysis_config, **kwargs,
    )


def test_lock_contract_invalid_settings_rejected(tmp_path):
    base = json.loads((ROOT / "configs/final_evaluation.json").read_text())
    for changes in (
        {"primary_split": "random_grouped"},
        {"primary_model": "random_forest_descriptors"},
        {"ridge_alpha": 0},
        {"test_access_policy": "allow_retuning"},
        {"expected_test_rows_per_split": 0},
        {"analysis_report_sha256": "0"},
    ):
        obj = {**base, **changes}
        file = tmp_path / "bad.json"
        file.write_text(json.dumps(obj))
        with pytest.raises(FinalEvaluationError, match="Invalid final evaluation lock"):
            load_lock(file)


def test_real_development_report_freeze_is_recorded():
    lock = load_lock(ROOT / "configs/final_evaluation.json")
    assert lock["primary_model"] == "ridge_morgan"
    assert lock["primary_split"] == "scaffold_held_out"
    assert lock["cv_selected_alpha_by_split"] == {
        "scaffold_held_out": 10.0, "random_grouped": 10.0,
    }
    assert lock["expected_test_rows_per_split"] == 420


def test_final_run_scores_test_once_and_keeps_source_unchanged(locked_toy, tmp_path):
    processed, baseline, analysis_dir, _, lock_file = locked_toy
    originals = {p: p.read_bytes() for p in [
        processed / "features.npz", processed / "scaffold_held_out.csv",
        baseline / "training_report.json", analysis_dir / "analysis_report.json",
        lock_file,
    ]}
    out = tmp_path / "one_time_test"
    result = run(locked_toy, out)
    assert set(result["strategies"]) == set(SPLIT_STRATEGIES)
    assert result["primary_split"] == "scaffold_held_out"
    assert result["primary_model"] == "ridge_morgan"
    _, arrays, assignments = load_prepared(processed)
    for strategy, detail in result["strategies"].items():
        test_records = [r for r in assignments[strategy] if r["split"] == "test"]
        test_identifiers = {r["csv_row"] for r in test_records}
        with (out / strategy / "test_predictions.csv").open(newline="", encoding="utf-8") as fp:
            predictions = list(csv.DictReader(fp))
        assert {r["csv_row"] for r in predictions} == test_identifiers
        assert detail["test_rows"] == len(predictions)
        assert set(detail["model_metrics"]) == {
            "dummy_mean", "ridge_descriptors", "ridge_morgan", "random_forest_descriptors"
        }
        metrics = detail["model_metrics"]["ridge_morgan"]
        observed = np.array([float(r["observed_logd"]) for r in predictions])
        predicted = np.array([float(r["ridge_morgan_prediction"]) for r in predictions])
        assert metrics["rmse"] == pytest.approx(float(np.sqrt(np.mean((observed-predicted)**2))), abs=2e-5)
        assert 0 <= detail["prediction_interval"]["empirical_test_coverage"] <= 1
        assert (out / strategy / "test_observed_vs_predicted.png").is_file()
        assert (out / strategy / "test_similarity_vs_error.png").is_file()
        assert detail["prediction_csv_sha256"] == digest(out / strategy / "test_predictions.csv")
    assert (out / "final_report.json").is_file()
    for path, data in originals.items():
        assert path.read_bytes() == data
    with pytest.raises(FinalEvaluationError, match="already exists"):
        run(locked_toy, out)


def test_wrong_lock_checksum_rejected_without_test_access(locked_toy, tmp_path):
    orig = json.loads(locked_toy[-1].read_text())
    orig["analysis_report_sha256"] = "a" * 64
    lock_file = tmp_path / "modified.json"
    lock_file.write_text(json.dumps(orig))
    processed, baseline, analysis, config, _ = locked_toy
    out = tmp_path / "must_not_exist"
    with pytest.raises(FinalEvaluationError, match="analysis report differs"):
        final_evaluation(processed, baseline, analysis, out, lock_file, ROOT / "configs/models.json", config)
    assert not out.exists()


def test_wrong_sklearn_version_rejected_before_deserialization(locked_toy, tmp_path):
    orig = json.loads(locked_toy[-1].read_text())
    orig["expected_sklearn_version"] = "0.0.0"
    f = tmp_path / "wrong_sklearn.json"
    f.write_text(json.dumps(orig))
    processed, baseline, analysis, config, _ = locked_toy
    out = tmp_path / "must_not_exist"
    with pytest.raises(FinalEvaluationError, match="saved models used scikit-learn|Saved models used scikit-learn"):
        final_evaluation(processed, baseline, analysis, out, f, ROOT / "configs/models.json", config)
    assert not out.exists()


def test_trusted_model_failing_validation_signature_is_blocked(locked_toy, tmp_path):
    processed, baseline, analysis, config, lock_file = locked_toy
    copy = tmp_path / "modified_models"
    shutil.copytree(baseline, copy)
    # Swapping in a different baseline estimator should fail on validation,
    # *before* the test partition is indexed.
    target = copy / "scaffold_held_out" / "ridge_morgan.joblib"
    shutil.copyfile(copy / "scaffold_held_out" / "dummy_mean.joblib", target)
    with pytest.raises(FinalEvaluationError, match="does not reproduce frozen validation"):
        final_evaluation(processed, copy, analysis, tmp_path / "no_result", lock_file,
                         ROOT / "configs/models.json", config)
    assert not (tmp_path / "no_result").exists()


def test_no_retraining_or_hidden_test_selection_in_module():
    import inspect
    from molprop import final_evaluation as module
    source = inspect.getsource(module)
    assert ".fit(" not in source
    assert "GridSearchCV" not in source
    assert "model_selection" not in source
