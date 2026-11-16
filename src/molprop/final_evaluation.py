"""One-shot evaluation of frozen, locally fitted molecular logD models.

Preflight verifies dataset/experiment hashes, split integrity, saved validation
predictions and interval calibration. Only after all checks pass are held-out
TEST rows selected. There is no model fit, tuning or model selection here.

Security: joblib files are pickle-based. Use this only with model artifacts you
created locally and trust. Never load downloaded or unsolicited joblib files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import __version__ as sklearn_version

from molprop.analyze import (
    _read_baselines,
    _subgroup_summary,
    grouped_bootstrap_rmse_difference,
    nearest_train_tanimoto,
)
from molprop.prepare import digest
from molprop.train import MODEL_NAMES, SPLIT_STRATEGIES, _metrics, load_model_config, load_prepared


class FinalEvaluationError(ValueError):
    """The locked final evaluation cannot safely proceed."""


def load_lock(path: Path) -> dict[str, Any]:
    """Require a complete, explicit, immutable-before-use protocol contract."""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or obj["protocol_version"] != 1:
            raise ValueError("protocol_version must be 1")
        if obj["primary_split"] != "scaffold_held_out" or obj["secondary_split"] != "random_grouped":
            raise ValueError("primary and secondary split definitions must be fixed")
        if obj["primary_model"] != "ridge_morgan" or obj["comparison_models"] != [
            "dummy_mean", "ridge_descriptors", "random_forest_descriptors"
        ]:
            raise ValueError("model comparison must be frozen")
        if obj["interval_model"] != "calibration_ridge_morgan":
            raise ValueError("calibration model must be frozen")
        if obj["test_access_policy"] != "single_final_evaluation_no_retraining_or_selection":
            raise ValueError("test-access policy mismatch")
        if not isinstance(obj["expected_test_rows_per_split"], int) or obj["expected_test_rows_per_split"] < 2:
            raise ValueError("invalid test-set size")
        for key in ("ridge_alpha", "nominal_interval_coverage"):
            v = obj[key]
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError(f"invalid {key}")
        if obj["ridge_alpha"] <= 0 or not 0 < obj["nominal_interval_coverage"] < 1:
            raise ValueError("invalid model or coverage setting")
        alphas = obj["cv_selected_alpha_by_split"]
        if set(alphas) != set(SPLIT_STRATEGIES) or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
            for v in alphas.values()
        ):
            raise ValueError("invalid CV-selected alpha contract")
        edges = obj["similarity_bins"]
        if (not isinstance(edges, list) or len(edges) < 3 or edges[0] != 0 or edges[-1] != 1
                or any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in edges)
                or any(a >= b for a, b in zip(edges, edges[1:]))):
            raise ValueError("invalid similarity bins")
        if (not isinstance(obj["bootstrap_repetitions"], int) or obj["bootstrap_repetitions"] < 100
                or not isinstance(obj["bootstrap_seed"], int)):
            raise ValueError("invalid bootstrap configuration")
        for key in (
            "preparation_report_sha256", "training_report_sha256", "analysis_report_sha256",
            "prepared_features_sha256", "model_config_sha256", "analysis_config_sha256",
        ):
            h = obj[key]
            if not isinstance(h, str) or len(h) != 64 or any(x not in "0123456789abcdef" for x in h):
                raise ValueError(f"invalid SHA-256 checksum {key}")
        if not isinstance(obj["expected_sklearn_version"], str) or not obj["expected_sklearn_version"]:
            raise ValueError("missing saved scikit-learn version")
        return obj
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise FinalEvaluationError(f"Invalid final evaluation lock {path}: {exc}") from exc


def _require_digest(path: Path, value: str, label: str) -> None:
    if not path.is_file():
        raise FinalEvaluationError(f"Missing {label}: {path}")
    if digest(path) != value:
        raise FinalEvaluationError(f"{label} differs from frozen protocol: {path}")


def _predict(model: Any, matrix: Any, *, name: str) -> np.ndarray:
    try:
        values = np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise FinalEvaluationError(f"Prediction failed for {name}: {exc}") from exc
    if values.shape != (matrix.shape[0],) or not np.isfinite(values).all():
        raise FinalEvaluationError(f"Nonfinite or misaligned predictions from {name}")
    return values


def _validated_model(path: Path, validation_features: Any, reference: np.ndarray, *, label: str) -> Any:
    if not path.is_file():
        raise FinalEvaluationError(f"Missing locally trained model: {path}")
    # ONLY trusted local artifacts. Joblib can execute code when loading.
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise FinalEvaluationError(f"Cannot load trusted local model {path}: {exc}") from exc
    predicted = _predict(model, validation_features, name=label)
    # Reference CSVs have six decimals. Permit tiny numerical/solver drift,
    # but reject changes that could reflect a different trained model.
    if not np.allclose(predicted, reference, rtol=0, atol=3e-4):
        maximum = float(np.max(np.abs(predicted - reference)))
        raise FinalEvaluationError(
            f"Stored {label} does not reproduce frozen validation predictions (max error {maximum:.6g})"
        )
    return model


def _read_validation_diagnostics(path: Path, expected: list[dict[str, str]], y_validation: np.ndarray):
    required = ["csv_row", "compound_id", "observed_logd", "interval_prediction", "interval_lower", "interval_upper"]
    try:
        with Path(path).open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, strict=True)
            if not reader.fieldnames or not set(required) <= set(reader.fieldnames):
                raise FinalEvaluationError("Analysis diagnostics missing expected interval columns")
            rows = list(reader)
    except OSError as exc:
        raise FinalEvaluationError(f"Missing analysis validation diagnostics: {path}") from exc
    if len(rows) != len(expected):
        raise FinalEvaluationError("Analysis validation row count mismatch")
    pred = np.empty(len(rows))
    low = np.empty(len(rows))
    high = np.empty(len(rows))
    for i, (got, src) in enumerate(zip(rows, expected)):
        if (got["csv_row"] != src["csv_row"] or got["compound_id"] != src["compound_id"]
                or abs(float(got["observed_logd"]) - y_validation[i]) > 1e-5):
            raise FinalEvaluationError("Analysis validation records disagree with the original split")
        pred[i], low[i], high[i] = (float(got[k]) for k in required[-3:])
    if not all(np.isfinite(x).all() for x in (pred, low, high)):
        raise FinalEvaluationError("Nonfinite reference interval values")
    return pred, low, high


def _preflight(
    processed_dir: Path, baseline_dir: Path, analysis_dir: Path, models_config: Path,
    analysis_config: Path, lock_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, list[dict[str, str]]], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate everything against development artifacts before TEST access."""
    lock = load_lock(lock_path)
    required = [
        (processed_dir / "preparation_report.json", "preparation_report_sha256", "preparation report"),
        (baseline_dir / "training_report.json", "training_report_sha256", "baseline training report"),
        (analysis_dir / "analysis_report.json", "analysis_report_sha256", "analysis report"),
        (models_config, "model_config_sha256", "model config"),
        (analysis_config, "analysis_config_sha256", "analysis config"),
    ]
    for path, key, label in required:
        _require_digest(path, lock[key], label)
    if sklearn_version != lock["expected_sklearn_version"]:
        raise FinalEvaluationError(
            f"Saved models used scikit-learn {lock['expected_sklearn_version']}, "
            f"but this environment has {sklearn_version}. Install the saved-model version."
        )
    source, arrays, assignments = load_prepared(processed_dir)
    if source["dataset"] != lock["dataset"]:
        raise FinalEvaluationError("Prepared dataset name differs from frozen protocol")
    if source["data_sha256"]["features.npz"] != lock["prepared_features_sha256"]:
        raise FinalEvaluationError("Prepared fingerprints differ from frozen dataset")
    model_cfg = load_model_config(models_config)
    if float(model_cfg["ridge_alpha_morgan"]) != lock["ridge_alpha"]:
        raise FinalEvaluationError("Primary model hyperparameter differs from frozen protocol")
    training = json.loads((baseline_dir / "training_report.json").read_text(encoding="utf-8"))
    analysis = json.loads((analysis_dir / "analysis_report.json").read_text(encoding="utf-8"))
    for report in (training, analysis):
        if report.get("test_evaluated") is not False or report.get("test_predictions_written") is not False:
            raise FinalEvaluationError("Development report indicates prior test evaluation")
        if report.get("sklearn_version") != lock["expected_sklearn_version"]:
            raise FinalEvaluationError("Development report scikit-learn version does not match lock")
        if report.get("preparation_report_sha256") != lock["preparation_report_sha256"]:
            raise FinalEvaluationError("Development report preparation checksum mismatch")
        if report.get("model_config_sha256") != lock["model_config_sha256"]:
            raise FinalEvaluationError("Development report model config checksum mismatch")
        if report.get("prepared_features_sha256") != lock["prepared_features_sha256"]:
            raise FinalEvaluationError("Development report feature checksum mismatch")
    if analysis.get("baseline_training_report_sha256") != lock["training_report_sha256"]:
        raise FinalEvaluationError("Analysis is not tied to the frozen baseline report")
    if analysis.get("analysis_config_sha256") != lock["analysis_config_sha256"]:
        raise FinalEvaluationError("Analysis config checksum mismatch")
    if analysis.get("selection_policy", "").find("No test access") == -1:
        raise FinalEvaluationError("Prior analysis does not confirm no test access")
    for strategy in SPLIT_STRATEGIES:
        if strategy not in analysis.get("strategies", {}) or strategy not in training.get("models", {}):
            raise FinalEvaluationError(f"Development report missing split {strategy}")
        s = analysis["strategies"][strategy]
        if s["cv_training_only"]["selected_alpha"] != lock["cv_selected_alpha_by_split"][strategy]:
            raise FinalEvaluationError("CV-selected model setting differs from frozen decision")
        interval = s["prediction_interval"]
        if (interval["model_alpha"] != lock["ridge_alpha"]
                or interval["nominal_coverage"] != lock["nominal_interval_coverage"]
                or interval["method"] != "split conformal absolute-residual interval"):
            raise FinalEvaluationError("Prediction interval settings differ from validation")
        if not math.isfinite(interval["quantile_logd"]) or interval["quantile_logd"] <= 0:
            raise FinalEvaluationError("Invalid saved calibration residual quantile")
    models_by_split: dict[str, dict[str, Any]] = {}
    for strategy in SPLIT_STRATEGIES:
        records = assignments[strategy]
        splits = np.array([record["split"] for record in records])
        val_idx = np.flatnonzero(splits == "validation")
        train_idx = np.flatnonzero(splits == "train")
        # IMPORTANT: still no TEST index or TEST label slice in this function.
        val_records = [records[int(i)] for i in val_idx]
        y_val = arrays["logd"][val_idx]
        reference = _read_baselines(baseline_dir, strategy, val_records, y_val, training)
        models: dict[str, Any] = {}
        for name in MODEL_NAMES:
            feature_type = model_cfg["representations"][name]
            features = arrays["morgan"] if feature_type == "morgan" else arrays["descriptors"]
            val_x = csr_matrix(features[val_idx], dtype=np.float64) if feature_type == "morgan" else features[val_idx]
            path = baseline_dir / strategy / f"{name}.joblib"
            models[name] = _validated_model(path, val_x, reference[name], label=f"{strategy}/{name}")
        interval_details = analysis["strategies"][strategy]["prediction_interval"]
        q = float(interval_details["quantile_logd"])
        cal_model = _validated_model(
            analysis_dir / strategy / "calibration_ridge_morgan.joblib",
            csr_matrix(arrays["morgan"][val_idx], dtype=np.float64),
            _read_validation_diagnostics(
                analysis_dir / strategy / "validation_diagnostics.csv", val_records, y_val
            )[0], label=f"{strategy}/calibration_ridge_morgan",
        )
        diag_pred, diag_low, diag_high = _read_validation_diagnostics(
            analysis_dir / strategy / "validation_diagnostics.csv", val_records, y_val
        )
        if (not np.allclose(diag_low, diag_pred - q, atol=3e-5, rtol=0)
                or not np.allclose(diag_high, diag_pred + q, atol=3e-5, rtol=0)):
            raise FinalEvaluationError("Saved validation intervals do not match the frozen calibration quantile")
        # Verify trained model shapes by predictions on existing validation rows.
        models["calibration_ridge_morgan"] = cal_model
        models_by_split[strategy] = models
    return lock, arrays, assignments, training, analysis, models_by_split


def _save_plot(folder: Path, truth: np.ndarray, predicted: np.ndarray, similarity: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(truth, predicted, s=14, alpha=0.6)
    lo, hi = min(truth.min(), predicted.min()), max(truth.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set(xlabel="Experimental logD", ylabel="Predicted logD", title="Frozen model: held-out test")
    fig.tight_layout()
    fig.savefig(folder / "test_observed_vs_predicted.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(similarity, np.abs(predicted - truth), s=14, alpha=0.6)
    ax.set(xlabel="Max training Morgan Tanimoto", ylabel="Absolute error (logD)",
           title="Held-out test error and training similarity")
    fig.tight_layout()
    fig.savefig(folder / "test_similarity_vs_error.png", dpi=140)
    plt.close(fig)


def final_evaluation(
    processed_dir: Path = Path("data/processed"),
    baseline_dir: Path = Path("outputs/baselines"),
    analysis_dir: Path = Path("outputs/analysis"),
    output_dir: Path = Path("outputs/final_evaluation"),
    lock_path: Path = Path("configs/final_evaluation.json"),
    model_config: Path = Path("configs/models.json"),
    analysis_config: Path = Path("configs/analysis.json"),
) -> dict[str, Any]:
    """Evaluate each saved model on each reserved TEST partition exactly once.

    This code has no `.fit` or parameter selection. All development/model
    verification precedes selecting a TEST index or accessing a TEST label.
    """
    processed_dir, baseline_dir, analysis_dir, output_dir = map(
        Path, (processed_dir, baseline_dir, analysis_dir, output_dir)
    )
    if output_dir.exists():
        raise FinalEvaluationError(
            f"Final evaluation output already exists: {output_dir}. "
            "Refusing to overwrite, retest or conceal a previous final result."
        )
    if output_dir.resolve() in (processed_dir.resolve(), baseline_dir.resolve(), analysis_dir.resolve()):
        raise FinalEvaluationError("Final report must be in a separate directory")
    lock, arrays, assignments, training, analysis, models = _preflight(
        processed_dir, baseline_dir, analysis_dir, model_config, analysis_config, lock_path
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Create in parent then atomically move finished report into its final path.
    with tempfile.TemporaryDirectory(prefix=".final-evaluation-", dir=output_dir.parent) as tmp:
        stage = Path(tmp)
        report: dict[str, Any] = {
            "dataset": lock["dataset"],
            "protocol_lock_sha256": digest(lock_path),
            "preparation_report_sha256": lock["preparation_report_sha256"],
            "training_report_sha256": lock["training_report_sha256"],
            "analysis_report_sha256": lock["analysis_report_sha256"],
            "sklearn_version": sklearn_version,
            "protocol": "One-time held-out test; settings frozen before test evaluation",
            "primary_split": lock["primary_split"],
            "primary_model": lock["primary_model"],
            "strategies": {},
            "warning": "Scientific ML benchmark only. A held-out score does not establish prospective clinical or drug-discovery performance.",
        }
        for strategy in SPLIT_STRATEGIES:
            records = assignments[strategy]
            parts = np.array([r["split"] for r in records])
            train_ix = np.flatnonzero(parts == "train")
            test_ix = np.flatnonzero(parts == "test")  # First permitted test access after preflight.
            if len(test_ix) != lock["expected_test_rows_per_split"]:
                raise FinalEvaluationError("Test partition size no longer matches the frozen protocol")
            train_records = [records[int(i)] for i in train_ix]
            test_records = [records[int(i)] for i in test_ix]
            if set(r["canonical_smiles"] for r in train_records) & set(r["canonical_smiles"] for r in test_records):
                raise FinalEvaluationError("Canonical molecular identity leakage across test boundary")
            if strategy == "scaffold_held_out" and set(r["scaffold"] for r in train_records) & set(r["scaffold"] for r in test_records):
                raise FinalEvaluationError("Scaffold leakage across test boundary")
            test_y = arrays["logd"][test_ix]
            predicted: dict[str, np.ndarray] = {}
            for name in MODEL_NAMES:
                rep = training["models"][strategy]["candidates_by_validation_rmse"]
                entry = next((m for m in rep if m["model"] == name), None)
                if entry is None:
                    raise FinalEvaluationError("Frozen baseline model not in training report")
                feature = arrays["morgan"] if entry["representation"] == "morgan" else arrays["descriptors"]
                x = csr_matrix(feature[test_ix], dtype=np.float64) if entry["representation"] == "morgan" else feature[test_ix]
                predicted[name] = _predict(models[strategy][name], x, name=f"{strategy}/{name}")
            cal_x = csr_matrix(arrays["morgan"][test_ix], dtype=np.float64)
            conformal_pred = _predict(
                models[strategy]["calibration_ridge_morgan"], cal_x, name=f"{strategy}/calibration"
            )
            q = analysis["strategies"][strategy]["prediction_interval"]["quantile_logd"]
            low, high = conformal_pred - q, conformal_pred + q
            coverage = float(np.mean((test_y >= low) & (test_y <= high)))
            model_metrics = {name: _metrics(test_y, predicted[name]) for name in MODEL_NAMES}
            similarity, neighbor = nearest_train_tanimoto(
                arrays["morgan"][train_ix], arrays["morgan"][test_ix],
                [row["csv_row"] for row in train_records],
            )
            bounds = lock["similarity_bins"]
            labels = [f"[{bounds[j]:g},{bounds[j+1]:g}{']' if j==len(bounds)-2 else ')'}"
                      for j in range(len(bounds)-1)]
            primary = predicted[lock["primary_model"]]
            bins = _subgroup_summary(similarity, test_y, primary, bounds, labels)
            # Training-only quantiles used to label test subgroups; no target-driven selection.
            quartiles = np.quantile(arrays["logd"][train_ix], [0.25, 0.5, 0.75])
            quartile_stats = _subgroup_summary(
                test_y, test_y, primary, [-math.inf, *quartiles, math.inf],
                [f"train-target-Q{j}" for j in range(1, 5)]
            )
            bootstrap = grouped_bootstrap_rmse_difference(
                test_y, primary, predicted["random_forest_descriptors"],
                [r["scaffold"] if strategy == "scaffold_held_out" else r["canonical_smiles"] for r in test_records],
                n_boot=lock["bootstrap_repetitions"], seed=lock["bootstrap_seed"],
            )
            out = stage / strategy
            out.mkdir()
            with (out / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
                columns = [
                    "csv_row", "compound_id", "observed_logd", *[f"{name}_prediction" for name in MODEL_NAMES],
                    "max_train_tanimoto", "nearest_train_csv_row", "calibration_prediction",
                    "interval_lower", "interval_upper",
                ]
                writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                for i, row in enumerate(test_records):
                    writer.writerow({
                        "csv_row": row["csv_row"], "compound_id": row["compound_id"],
                        "observed_logd": f"{test_y[i]:.6f}",
                        **{f"{name}_prediction": f"{predicted[name][i]:.6f}" for name in MODEL_NAMES},
                        "max_train_tanimoto": f"{similarity[i]:.6f}",
                        "nearest_train_csv_row": neighbor[i],
                        "calibration_prediction": f"{conformal_pred[i]:.6f}",
                        "interval_lower": f"{low[i]:.6f}", "interval_upper": f"{high[i]:.6f}",
                    })
            _save_plot(out, test_y, primary, similarity)
            report["strategies"][strategy] = {
                "split_role": "primary" if strategy == lock["primary_split"] else "secondary",
                "test_rows": len(test_ix),
                "model_metrics": model_metrics,
                "similarity": {
                    "mean": round(float(similarity.mean()), 6),
                    "median": round(float(np.median(similarity)), 6),
                    "error_by_similarity_bin": bins,
                },
                "error_by_train_target_quartile": {
                    "cutpoints": [round(float(x), 6) for x in quartiles],
                    "bins": quartile_stats,
                },
                "ridge_minus_forest_rmse_bootstrap": bootstrap,
                "prediction_interval": {
                    "model": "separately fitted calibration Ridge (not the 3360-row Ridge baseline)",
                    "nominal_coverage": lock["nominal_interval_coverage"],
                    "empirical_test_coverage": round(coverage, 6),
                    "mean_width_logd": round(float(np.mean(high - low)), 6),
                    "calibration_quantile_logd": q,
                    "observations_covered": int(np.sum((test_y >= low) & (test_y <= high))),
                    "test_rows": len(test_y),
                    "warning": "Conformal coverage may fail under scaffold/chemical distribution shift; no prospective guarantee.",
                },
                "model_artifact_sha256": {
                    name: digest(baseline_dir / strategy / f"{name}.joblib") for name in MODEL_NAMES
                } | {"calibration_ridge_morgan": digest(analysis_dir / strategy / "calibration_ridge_morgan.joblib")},
                "prediction_csv_sha256": digest(out / "test_predictions.csv"),
            }
        (stage / "final_report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        # No overwrite: os.rename on Windows may refuse if destination exists.
        if output_dir.exists():
            raise FinalEvaluationError("A final report appeared during evaluation; refusing overwrite")
        os.rename(stage, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen, one-time held-out molecular logD evaluation")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_evaluation"))
    parser.add_argument("--lock", type=Path, default=Path("configs/final_evaluation.json"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--analysis-config", type=Path, default=Path("configs/analysis.json"))
    args = parser.parse_args()
    try:
        result = final_evaluation(
            args.processed_dir, args.baseline_dir, args.analysis_dir, args.output_dir,
            args.lock, args.models_config, args.analysis_config,
        )
    except (FinalEvaluationError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
