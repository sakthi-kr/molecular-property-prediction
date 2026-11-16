"""Train leakage-aware lipophilicity baselines using train and validation only.

No estimator is fitted or evaluated on the held-out test subset. Training
features and split assignments are loaded from checksummed Phase-2 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import __version__ as sklearn_version
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from molprop.prepare import digest
from molprop.splitting import SPLITS, audit_assignments

SPLIT_STRATEGIES = ("random_grouped", "scaffold_held_out")
MODEL_NAMES = ("dummy_mean", "ridge_descriptors", "ridge_morgan", "random_forest_descriptors")


class TrainingError(ValueError):
    """Prepared files, split contracts or fixed model settings are invalid."""


def load_model_config(path: Path) -> dict[str, Any]:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        assert isinstance(cfg, dict)
        assert isinstance(cfg["seed"], int) and not isinstance(cfg["seed"], bool)
        assert cfg["forest_n_estimators"] >= 10 and isinstance(cfg["forest_n_estimators"], int)
        assert cfg["forest_min_samples_leaf"] >= 1 and isinstance(cfg["forest_min_samples_leaf"], int)
        assert 0 < float(cfg["forest_max_features"]) <= 1
        assert cfg["forest_n_jobs"] >= 1 and isinstance(cfg["forest_n_jobs"], int)
        for name in ("ridge_alpha_descriptors", "ridge_alpha_morgan"):
            assert math.isfinite(float(cfg[name])) and float(cfg[name]) > 0
        assert cfg["representations"] == {
            "dummy_mean": "descriptors", "ridge_descriptors": "descriptors",
            "ridge_morgan": "morgan", "random_forest_descriptors": "descriptors",
        }
        return cfg
    except (OSError, AssertionError, ValueError, KeyError, TypeError) as exc:
        raise TrainingError(f"Invalid fixed model configuration {path}: {exc}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, strict=True)
        if list(reader.fieldnames or []) != ["csv_row", "compound_id", "canonical_smiles", "scaffold", "split"]:
            raise TrainingError(f"Unexpected split columns in {path.name}")
        return list(reader)


def load_prepared(processed_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, list[dict[str, str]]]]:
    """Verify provenance, content, indices, and no group overlap before training."""
    folder = Path(processed_dir)
    report_path = folder / "preparation_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for name in ("features.npz", "random_grouped.csv", "scaffold_held_out.csv"):
            if digest(folder / name) != report["data_sha256"][name]:
                raise TrainingError(f"Checksum mismatch for {name}; rerun preparation")
        with np.load(folder / "features.npz", allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]).copy() for key in ("descriptors", "morgan", "logd", "csv_row")}
        count = report["rows"]
        if not isinstance(count, int) or count < 15:
            raise TrainingError("Not enough prepared molecules")
        if arrays["descriptors"].shape != (count, report["descriptor_count"]):
            raise TrainingError("Descriptor array shape mismatch")
        if arrays["morgan"].shape != (count, report["morgan_fingerprint"]["bits"]):
            raise TrainingError("Fingerprint array shape mismatch")
        if arrays["logd"].shape != (count,) or arrays["csv_row"].shape != (count,):
            raise TrainingError("Targets or row identifiers have wrong shape")
        if not np.isfinite(arrays["descriptors"]).all() or not np.isfinite(arrays["logd"]).all():
            raise TrainingError("Nonfinite prepared numeric data")
        if not np.isin(arrays["morgan"], [0, 1]).all():
            raise TrainingError("Fingerprint matrix must contain bits only")
        if len(set(arrays["csv_row"].tolist())) != count:
            raise TrainingError("Repeated CSV row identifiers")
        assignments = {strategy: _read_csv(folder / f"{strategy}.csv") for strategy in SPLIT_STRATEGIES}
        for strategy, rows in assignments.items():
            if len(rows) != count:
                raise TrainingError(f"{strategy} split length mismatch")
            if [int(r["csv_row"]) for r in rows] != arrays["csv_row"].tolist():
                raise TrainingError(f"{strategy} split ordering does not match features")
            parts = [r["split"] for r in rows]
            if not set(parts) <= set(SPLITS):
                raise TrainingError(f"Unknown split label in {strategy}")
            checks = audit_assignments([r["canonical_smiles"] for r in rows], [r["scaffold"] for r in rows], parts)
            if checks != report["splits"][strategy]:
                raise TrainingError(f"{strategy} split audit changed since preparation")
            if strategy == "scaffold_held_out" and checks["scaffolds_crossing_partitions"]:
                raise TrainingError("Scaffold leakage detected")
            for part in ("train", "validation"):
                if parts.count(part) < 2:
                    raise TrainingError(f"{strategy} {part} has too few rows to evaluate")
        if [r["canonical_smiles"] for r in assignments[SPLIT_STRATEGIES[0]]] != [r["canonical_smiles"] for r in assignments[SPLIT_STRATEGIES[1]]]:
            raise TrainingError("Split files describe different molecular sequences")
        return report, arrays, assignments
    except (FileNotFoundError, KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(f"Prepared artifact verification failed: {exc}") from exc


def _candidate(name: str, cfg: dict[str, Any]):
    if name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge_descriptors":
        # The scaler learns its mean and scale on the TRAIN partition only.
        return make_pipeline(StandardScaler(), Ridge(alpha=float(cfg["ridge_alpha_descriptors"])))
    if name == "ridge_morgan":
        return Ridge(alpha=float(cfg["ridge_alpha_morgan"]), solver="lsqr")
    if name == "random_forest_descriptors":
        return RandomForestRegressor(
            n_estimators=cfg["forest_n_estimators"], min_samples_leaf=cfg["forest_min_samples_leaf"],
            max_features=float(cfg["forest_max_features"]), n_jobs=cfg["forest_n_jobs"],
            random_state=cfg["seed"],
        )
    raise TrainingError(f"Unregistered model {name}")


def _metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if not np.isfinite(predicted).all():
        raise TrainingError("A model produced nonfinite predictions")
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, predicted))), 6),
        "mae": round(float(mean_absolute_error(y_true, predicted)), 6),
        "r2": round(float(r2_score(y_true, predicted)), 6),
    }


def fit_baselines(
    processed_dir: Path = Path("data/processed"),
    output_dir: Path = Path("outputs/baselines"),
    config_path: Path = Path("configs/models.json"),
    *, force: bool = False,
) -> dict[str, Any]:
    """Train four fixed candidates per split; never fit, predict, score or summarize test labels."""
    processed_dir, output_dir, config_path = map(Path, (processed_dir, output_dir, config_path))
    cfg = load_model_config(config_path)
    source_report, arrays, assignments = load_prepared(processed_dir)
    if output_dir.exists() and (output_dir / "training_report.json").exists() and not force:
        raise TrainingError("Model outputs already exist; use --force to replace the previous run")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".training-", dir=output_dir) as tmp:
        staging = Path(tmp)
        report: dict[str, Any] = {
            "dataset": source_report["dataset"], "rows": source_report["rows"],
            "preparation_report_sha256": digest(processed_dir / "preparation_report.json"),
            "prepared_features_sha256": source_report["data_sha256"]["features.npz"],
            "model_config_sha256": digest(config_path), "sklearn_version": sklearn_version,
            "models": {}, "selection_metric": "validation RMSE (lower is better)",
            "test_evaluated": False, "test_predictions_written": False,
            "note": "Fixed baseline models only; all preprocessing fitted on training data. No test evaluation or hyperparameter search.",
        }
        for strategy in SPLIT_STRATEGIES:
            rows = assignments[strategy]
            parts = np.asarray([row["split"] for row in rows], dtype="U12")
            train = np.flatnonzero(parts == "train")
            val = np.flatnonzero(parts == "validation")
            # We purposely never create a test index array or inspect test targets.
            labels = arrays["logd"]
            train_targets = labels[train]
            val_targets = labels[val]
            validation_records: list[dict[str, Any]] = []
            records: list[dict[str, Any]] = []
            strategy_dir = staging / strategy
            strategy_dir.mkdir()
            for name in MODEL_NAMES:
                feature_type = cfg["representations"][name]
                feature_values = arrays["descriptors"] if feature_type == "descriptors" else arrays["morgan"]
                # Convert fingerprints to sparse matrix to reduce model memory.
                x_train = (csr_matrix(feature_values[train], dtype=np.float64) if feature_type == "morgan" else feature_values[train])
                x_val = (csr_matrix(feature_values[val], dtype=np.float64) if feature_type == "morgan" else feature_values[val])
                candidate = _candidate(name, cfg)
                candidate.fit(x_train, train_targets)
                train_predictions = candidate.predict(x_train)
                val_predictions = candidate.predict(x_val)
                entry = {"model": name, "representation": feature_type,
                         "train": _metrics(train_targets, train_predictions),
                         "validation": _metrics(val_targets, val_predictions)}
                records.append(entry)
                joblib.dump(candidate, strategy_dir / f"{name}.joblib", compress=3)
                for j, index in enumerate(val):
                    validation_records.append({
                        "model": name, "csv_row": str(rows[index]["csv_row"]),
                        "compound_id": rows[index]["compound_id"],
                        "observed_logd": f"{val_targets[j]:.6f}",
                        "predicted_logd": f"{val_predictions[j]:.6f}",
                    })
            records.sort(key=lambda x: (x["validation"]["rmse"], x["model"]))
            report["models"][strategy] = {
                "train_rows": int(train.size), "validation_rows": int(val.size),
                "candidates_by_validation_rmse": records,
                "best_validation_model": records[0]["model"],
            }
            with (strategy_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=("model", "csv_row", "compound_id", "observed_logd", "predicted_logd"), lineterminator="\n")
                writer.writeheader()
                writer.writerows(validation_records)
        (staging / "training_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        # Replacing report last ensures an incomplete run cannot be mistaken for a verified report.
        for strategy in SPLIT_STRATEGIES:
            (output_dir / strategy).mkdir(exist_ok=True)
            for path in (staging / strategy).iterdir():
                os.replace(path, output_dir / strategy / path.name)
        os.replace(staging / "training_report.json", output_dir / "training_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit lipophilicity baselines on train/validation subsets without touching the test set")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--force", action="store_true", help="Explicitly replace earlier local model artifacts")
    args = parser.parse_args()
    try:
        result = fit_baselines(args.processed_dir, args.output_dir, args.config, force=args.force)
    except (TrainingError, OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
