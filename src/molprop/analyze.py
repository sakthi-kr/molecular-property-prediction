"""Validation-only molecular generalisation and uncertainty analysis.

All model fitting, group CV, and interval calibration use TRAIN rows only.
Original validation partitions are used only to report performance, similarity,
and interval coverage. Original TEST rows are never indexed or predicted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import DataStructs
from scipy.sparse import csr_matrix
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from molprop.prepare import digest
from molprop.train import (
    MODEL_NAMES, SPLIT_STRATEGIES, _metrics, load_model_config, load_prepared,
)


class AnalysisError(ValueError):
    """Input provenance or development-only analysis contract was violated."""


def load_analysis_config(path: Path) -> dict[str, Any]:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("JSON object required")
        seed = cfg["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        grid = cfg["ridge_morgan_alpha_grid"]
        if not isinstance(grid, list) or not grid or any(
            isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or a <= 0
            for a in grid
        ) or len(set(grid)) != len(grid):
            raise ValueError("ridge alpha grid must contain unique positive finite values")
        for key, minimum in (("group_cv_folds", 2), ("bootstrap_repetitions", 100), ("max_outliers", 1)):
            v = cfg[key]
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("prediction_interval_coverage", "training_calibration_group_fraction"):
            v = cfg[key]
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not 0 < v < 1:
                raise ValueError(f"{key} must be between zero and one")
        edges = cfg["similarity_bins"]
        if not isinstance(edges, list) or len(edges) < 3 or edges[0] != 0 or edges[-1] != 1 or any(
            not isinstance(v, (float, int)) or not math.isfinite(v) for v in edges
        ) or any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("similarity bins must increase strictly from 0 to 1")
        return cfg
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise AnalysisError(f"Invalid analysis config {path}: {exc}") from exc


def _read_baselines(base_dir: Path, strategy: str, val_rows: list[dict[str, str]],
                    val_targets: np.ndarray, report: dict[str, Any]) -> dict[str, np.ndarray]:
    path = base_dir / strategy / "validation_predictions.csv"
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != ["model", "csv_row", "compound_id", "observed_logd", "predicted_logd"]:
                raise AnalysisError(f"Unexpected baseline prediction columns in {path}")
            rows = list(reader)
    except OSError as exc:
        raise AnalysisError(f"Missing baseline prediction file: {path}") from exc
    val_id_to_pos = {int(r["csv_row"]): j for j, r in enumerate(val_rows)}
    if len(rows) != len(MODEL_NAMES) * len(val_rows):
        raise AnalysisError(f"Unexpected validation prediction count in {path}")
    predictions = {model: np.full(len(val_rows), np.nan, dtype=np.float64) for model in MODEL_NAMES}
    for row in rows:
        try:
            model = row["model"]
            i = val_id_to_pos[int(row["csv_row"])]
            observed = float(row["observed_logd"])
            predicted = float(row["predicted_logd"])
        except (ValueError, TypeError, KeyError) as exc:
            raise AnalysisError(f"Unexpected or non-validation row in {path}") from exc
        if model not in predictions or row["compound_id"] != val_rows[i]["compound_id"]:
            raise AnalysisError("Prediction model or compound ID does not match the split")
        if not (math.isfinite(observed) and math.isfinite(predicted)):
            raise AnalysisError("Nonfinite baseline predictions")
        if abs(observed - val_targets[i]) > 1e-5 or np.isfinite(predictions[model][i]):
            raise AnalysisError("Baseline prediction duplicates a row or disagrees with validation target")
        predictions[model][i] = predicted
    for model in MODEL_NAMES:
        if not np.isfinite(predictions[model]).all():
            raise AnalysisError(f"Missing predictions from {model}")
        recorded = next((x for x in report["models"][strategy]["candidates_by_validation_rmse"]
                         if x["model"] == model), None)
        if recorded is None:
            raise AnalysisError(f"Baseline report is missing {model}")
        recalculated = _metrics(val_targets, predictions[model])
        if any(abs(recalculated[key] - recorded["validation"][key]) > 0.00001
               for key in ("rmse", "mae", "r2")):
            raise AnalysisError(f"Baseline metrics disagree with saved predictions: {model}")
    return predictions


def load_verified_inputs(processed: Path, baselines: Path, models_config: Path):
    """Verify input artifacts before any analysis; no pickle/joblib is loaded."""
    source, arrays, assignments = load_prepared(processed)
    try:
        training_report_path = baselines / "training_report.json"
        report = json.loads(training_report_path.read_text(encoding="utf-8"))
        if report["preparation_report_sha256"] != digest(processed / "preparation_report.json"):
            raise AnalysisError("Baseline report does not match the preparation report")
        if report["prepared_features_sha256"] != source["data_sha256"]["features.npz"]:
            raise AnalysisError("Baseline report refers to a different feature archive")
        if report["model_config_sha256"] != digest(models_config):
            raise AnalysisError("Baseline report does not match the fixed model config")
        if report.get("test_evaluated") is not False or report.get("test_predictions_written") is not False:
            raise AnalysisError("This analysis requires a baseline report with untouched test data")
        load_model_config(models_config)
        for strategy in SPLIT_STRATEGIES:
            if strategy not in report["models"]:
                raise AnalysisError(f"Missing baseline summary for {strategy}")
        return source, arrays, assignments, report
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError(f"Baseline provenance check failed: {exc}") from exc


def _bits_to_rdkit(data: np.ndarray) -> list[DataStructs.ExplicitBitVect]:
    if data.ndim != 2 or not np.isin(data, [0, 1]).all():
        raise AnalysisError("Expected a binary fingerprint matrix")
    out = []
    for row in data:
        bv = DataStructs.ExplicitBitVect(int(len(row)))
        bv.SetBitsFromList(np.flatnonzero(row).tolist())
        out.append(bv)
    return out


def nearest_train_tanimoto(train_bits: np.ndarray, validation_bits: np.ndarray,
                            train_identifiers: list[str]) -> tuple[np.ndarray, list[str]]:
    """Max hashed Morgan Tanimoto per validation molecule versus training only."""
    if train_bits.ndim != 2 or validation_bits.ndim != 2 or train_bits.shape[1] != validation_bits.shape[1]:
        raise AnalysisError("Fingerprint width mismatch")
    if not len(train_bits) or len(train_identifiers) != len(train_bits):
        raise AnalysisError("Missing training fingerprints or identifiers")
    train = _bits_to_rdkit(train_bits)
    validation = _bits_to_rdkit(validation_bits)
    similarity = np.zeros(len(validation), dtype=np.float64)
    neighbor_ids = []
    for i, fp in enumerate(validation):
        sims = DataStructs.BulkTanimotoSimilarity(fp, train)
        j = int(np.argmax(sims))
        similarity[i] = float(sims[j])
        neighbor_ids.append(train_identifiers[j])
    return similarity, neighbor_ids


def grouped_cv_ridge(x_train: csr_matrix, y_train: np.ndarray, groups: np.ndarray,
                     alphas: list[float], folds: int) -> dict[str, Any]:
    unique = len(set(groups.tolist()))
    if unique < folds:
        raise AnalysisError(f"Need at least {folds} train groups; found {unique}")
    result = []
    cv = list(GroupKFold(n_splits=folds).split(x_train, y_train, groups))
    for fit, val in cv:
        if set(groups[fit]).intersection(set(groups[val])):
            raise AnalysisError("Group overlap in cross-validation fold")
    for a in alphas:
        fold_rmse = []
        for fit, val in cv:
            model = Ridge(alpha=float(a), solver="lsqr")
            model.fit(x_train[fit], y_train[fit])
            pred = model.predict(x_train[val])
            fold_rmse.append(float(np.sqrt(np.mean((y_train[val] - pred) ** 2))))
        result.append({"alpha": float(a), "fold_rmse": [round(v, 6) for v in fold_rmse],
                       "mean_rmse": round(float(np.mean(fold_rmse)), 6),
                       "std_rmse": round(float(np.std(fold_rmse)), 6)})
    winner = min(result, key=lambda r: (r["mean_rmse"], r["alpha"]))
    return {"group_count": unique, "folds": folds,
            "cv_metric": "mean fold RMSE", "candidates": result,
            "selected_alpha": winner["alpha"]}


def split_conformal_ridge(x_train: csr_matrix, y_train: np.ndarray, groups: np.ndarray,
                          x_val: csr_matrix, y_val: np.ndarray, *,
                          alpha: float, fraction: float, coverage: float, seed: int):
    """Fixed-alpha model; calibration excluded from fitting and tuning."""
    train_rel, cal_rel = next(GroupShuffleSplit(
        n_splits=1, test_size=fraction, random_state=seed
    ).split(x_train, y_train, groups=groups))
    if set(groups[train_rel]) & set(groups[cal_rel]):
        raise AnalysisError("Calibration group leakage")
    if len(train_rel) < 2 or len(cal_rel) < 2:
        raise AnalysisError("Not enough groups for proper-training/calibration separation")
    model = Ridge(alpha=float(alpha), solver="lsqr")
    model.fit(x_train[train_rel], y_train[train_rel])
    cal_predictions = model.predict(x_train[cal_rel])
    residuals = np.abs(y_train[cal_rel] - cal_predictions)
    # Conservative split-conformal order statistic. If too few calibration
    # observations, the mathematically correct bound is infinite; refuse it.
    rank = math.ceil((len(cal_rel) + 1) * coverage)
    if rank > len(cal_rel):
        raise AnalysisError("Too few calibration rows for requested finite-quantile coverage")
    q = float(np.partition(residuals, rank - 1)[rank - 1])
    predictions = model.predict(x_val)
    low, high = predictions - q, predictions + q
    summary = {"method": "split conformal absolute-residual interval",
               "model": "fixed-alpha Ridge Morgan, fitted on proper-training only",
               "model_alpha": float(alpha), "nominal_coverage": coverage,
               "training_rows": len(train_rel), "calibration_rows": len(cal_rel),
               "training_groups": len(set(groups[train_rel])),
               "calibration_groups": len(set(groups[cal_rel])),
               "quantile_rank": rank, "quantile_logd": round(q, 6),
               "validation_rows": len(y_val),
               "validation_empirical_coverage": round(float(np.mean((y_val >= low) & (y_val <= high))), 6),
               "validation_mean_interval_width": round(float(np.mean(high - low)), 6),
               "warning": "The nominal coverage is not guaranteed under scaffold/chemical distribution shift."}
    return summary, (predictions, low, high), model


def grouped_bootstrap_rmse_difference(y: np.ndarray, first: np.ndarray, second: np.ndarray,
                                      groups: list[str], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Paired cluster bootstrap; difference = RMSE(first) - RMSE(second)."""
    if len(y) != len(first) or len(y) != len(second) or len(y) != len(groups) or not len(y):
        raise AnalysisError("Bootstrapping requires aligned nonempty validation rows")
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        by_group[group].append(i)
    keys = sorted(by_group)
    index = [np.asarray(by_group[g], dtype=np.int64) for g in keys]
    counts = np.array([len(x) for x in index], dtype=np.float64)
    sq1 = (y - first) ** 2
    sq2 = (y - second) ** 2
    totals1 = np.array([sq1[ix].sum() for ix in index], dtype=np.float64)
    totals2 = np.array([sq2[ix].sum() for ix in index], dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(keys), size=len(keys))
        denom = counts[chosen].sum()
        diffs[b] = np.sqrt(totals1[chosen].sum() / denom) - np.sqrt(totals2[chosen].sum() / denom)
    observed = float(np.sqrt(sq1.mean()) - np.sqrt(sq2.mean()))
    return {"difference_definition": "RMSE(first) - RMSE(second)", "point_estimate": round(observed, 6),
            "cluster_type": "scaffold" if len(set(groups)) < len(groups) else "molecule or unique group",
            "groups": len(keys), "resamples": n_boot,
            "approx_95pct_interval": [round(float(x), 6) for x in np.quantile(diffs, [0.025, 0.975])],
            "warning": "Exploratory cluster bootstrap on one validation split; not an external performance guarantee."}


def _subgroup_summary(values: np.ndarray, observed: np.ndarray, predicted: np.ndarray,
                      edges: list[float], labels: list[str]):
    result = []
    for i, title in enumerate(labels):
        low, high = edges[i], edges[i + 1]
        membership = (values >= low) & ((values <= high) if i == len(labels) - 1 else (values < high))
        indices = np.flatnonzero(membership)
        if len(indices):
            diff = predicted[indices] - observed[indices]
            result.append({"bin": title, "n": int(len(indices)),
                           "mae": round(float(np.abs(diff).mean()), 6),
                           "bias_pred_minus_obs": round(float(diff.mean()), 6),
                           "rmse": round(float(np.sqrt(np.mean(diff ** 2))), 6)})
        else:
            result.append({"bin": title, "n": 0, "mae": None, "bias_pred_minus_obs": None, "rmse": None})
    return result


def _save_figures(folder: Path, y: np.ndarray, predicted: np.ndarray, similarity: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y, predicted, s=13, alpha=0.6)
    min_y, max_y = min(np.min(y), np.min(predicted)), max(np.max(y), np.max(predicted))
    ax.plot([min_y, max_y], [min_y, max_y], linestyle="--", linewidth=1)
    ax.set(xlabel="Experimental logD", ylabel="Predicted logD", title="Validation predictions")
    fig.tight_layout()
    fig.savefig(folder / "observed_vs_predicted.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(similarity, np.abs(predicted - y), s=13, alpha=0.55)
    ax.set(xlabel="Max train-set Morgan Tanimoto", ylabel="Absolute error (logD)",
           title="Validation error versus training similarity")
    fig.tight_layout()
    fig.savefig(folder / "similarity_vs_error.png", dpi=140)
    plt.close(fig)


def analyze(
    processed_dir: Path = Path("data/processed"),
    baseline_dir: Path = Path("outputs/baselines"),
    output_dir: Path = Path("outputs/analysis"),
    model_config: Path = Path("configs/models.json"),
    analysis_config: Path = Path("configs/analysis.json"),
    *, force: bool = False,
) -> dict[str, Any]:
    """Audit baseline errors and improve Ridge using train-group CV only."""
    processed_dir, baseline_dir, output_dir = map(Path, (processed_dir, baseline_dir, output_dir))
    config = load_analysis_config(analysis_config)
    source, arrays, assignments, baseline = load_verified_inputs(processed_dir, baseline_dir, model_config)
    if (output_dir / "analysis_report.json").exists() and not force:
        raise AnalysisError("Analysis outputs already exist; use --force explicitly to replace them")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".validation-analysis-", dir=output_dir) as tmp:
        stage = Path(tmp)
        report: dict[str, Any] = {
            "dataset": source["dataset"],
            "prepared_features_sha256": source["data_sha256"]["features.npz"],
            "preparation_report_sha256": digest(processed_dir / "preparation_report.json"),
            "baseline_training_report_sha256": digest(baseline_dir / "training_report.json"),
            "model_config_sha256": digest(model_config),
            "analysis_config_sha256": digest(analysis_config),
            "sklearn_version": sklearn_version,
            "test_evaluated": False, "test_predictions_written": False,
            "selection_policy": "Alpha selected by train-only group CV; validation reports a fixed candidate comparison. No test access.",
            "strategies": {},
        }
        for strategy in SPLIT_STRATEGIES:
            rows = assignments[strategy]
            labels = np.asarray([x["split"] for x in rows], dtype="U12")
            train_ix = np.flatnonzero(labels == "train")
            val_ix = np.flatnonzero(labels == "validation")
            if len(train_ix) < 10 or len(val_ix) < 2:
                raise AnalysisError("Training/validation partition too small")
            # Explicitly no test index or test-target slice anywhere in this module.
            ytrain, yval = arrays["logd"][train_ix], arrays["logd"][val_ix]
            train_rows = [rows[int(i)] for i in train_ix]
            val_rows = [rows[int(i)] for i in val_ix]
            grouping = "scaffold" if strategy == "scaffold_held_out" else "canonical_smiles"
            groups_train = np.array([r[grouping] for r in train_rows], dtype=object)
            groups_val = [r[grouping] for r in val_rows]
            xtrain = csr_matrix(arrays["morgan"][train_ix], dtype=np.float64)
            xval = csr_matrix(arrays["morgan"][val_ix], dtype=np.float64)
            base = _read_baselines(baseline_dir, strategy, val_rows, yval, baseline)
            cv = grouped_cv_ridge(xtrain, ytrain, groups_train, config["ridge_morgan_alpha_grid"], config["group_cv_folds"])
            tuned = Ridge(alpha=cv["selected_alpha"], solver="lsqr")
            tuned.fit(xtrain, ytrain)
            predictions = tuned.predict(xval)
            if not np.isfinite(predictions).all():
                raise AnalysisError("Nonfinite tuned predictions")
            tuned_metrics = _metrics(yval, predictions)
            similarity, neighbors = nearest_train_tanimoto(
                arrays["morgan"][train_ix], arrays["morgan"][val_ix],
                [r["csv_row"] for r in train_rows],
            )
            bounds = config["similarity_bins"]
            bin_labels = [f"[{bounds[j]:g},{bounds[j+1]:g}{']' if j==len(bounds)-2 else ')'}"
                          for j in range(len(bounds) - 1)]
            sim_bins = _subgroup_summary(similarity, yval, predictions, bounds, bin_labels)
            # All target quartile cut points are defined using TRAIN targets only.
            quantiles = np.quantile(ytrain, [0.25, 0.5, 0.75])
            quart_edges = [-math.inf, *quantiles.tolist(), math.inf]
            quart_labels = [f"train-target-Q{j}" for j in range(1, 5)]
            quart_stats = _subgroup_summary(yval, yval, predictions, quart_edges, quart_labels)
            bootstrap = grouped_bootstrap_rmse_difference(
                yval, predictions, base["ridge_morgan"], groups_val,
                n_boot=config["bootstrap_repetitions"], seed=config["seed"],
            )
            # Calibration-model hyperparameter is fixed before splitting;
            # no calibration labels are used to select this alpha.
            conformal, intervals, interval_model = split_conformal_ridge(
                xtrain, ytrain, groups_train, xval, yval,
                alpha=float(load_model_config(model_config)["ridge_alpha_morgan"]),
                fraction=config["training_calibration_group_fraction"],
                coverage=config["prediction_interval_coverage"], seed=config["seed"],
            )
            con_pred, lower, upper = intervals
            subdir = stage / strategy
            subdir.mkdir()
            joblib.dump(tuned, subdir / "cv_selected_ridge_morgan.joblib", compress=3)
            joblib.dump(interval_model, subdir / "calibration_ridge_morgan.joblib", compress=3)
            with (subdir / "validation_diagnostics.csv").open("w", newline="", encoding="utf-8") as output:
                columns = ["csv_row", "compound_id", "canonical_smiles", "scaffold", "observed_logd",
                           "baseline_ridge_logd", "baseline_forest_logd", "cv_ridge_logd", "abs_error_cv_ridge",
                           "max_train_tanimoto", "nearest_train_csv_row", "interval_prediction", "interval_lower", "interval_upper"]
                writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                for i, row in enumerate(val_rows):
                    writer.writerow({"csv_row": row["csv_row"], "compound_id": row["compound_id"],
                                     "canonical_smiles": row["canonical_smiles"], "scaffold": row["scaffold"],
                                     "observed_logd": f"{yval[i]:.6f}",
                                     "baseline_ridge_logd": f"{base['ridge_morgan'][i]:.6f}",
                                     "baseline_forest_logd": f"{base['random_forest_descriptors'][i]:.6f}",
                                     "cv_ridge_logd": f"{predictions[i]:.6f}",
                                     "abs_error_cv_ridge": f"{abs(predictions[i]-yval[i]):.6f}",
                                     "max_train_tanimoto": f"{similarity[i]:.6f}",
                                     "nearest_train_csv_row": neighbors[i],
                                     "interval_prediction": f"{con_pred[i]:.6f}",
                                     "interval_lower": f"{lower[i]:.6f}",
                                     "interval_upper": f"{upper[i]:.6f}"})
            _save_figures(subdir, yval, predictions, similarity)
            worst_idx = np.argsort(-np.abs(yval - predictions), kind="stable")[:config["max_outliers"]]
            report["strategies"][strategy] = {
                "train_rows": int(len(train_ix)), "validation_rows": int(len(val_ix)),
                "cv_training_only": cv,
                "baseline_ridge_morgan_validation": _metrics(yval, base["ridge_morgan"]),
                "baseline_forest_descriptors_validation": _metrics(yval, base["random_forest_descriptors"]),
                "cv_selected_ridge_morgan_validation": tuned_metrics,
                "paired_cluster_bootstrap_tuned_minus_baseline_ridge": bootstrap,
                "similarity": {"metric": "max RDKit Morgan/Tanimoto against training partition only",
                               "mean": round(float(similarity.mean()), 6),
                               "median": round(float(np.median(similarity)), 6),
                               "min": round(float(similarity.min()), 6),
                               "max": round(float(similarity.max()), 6),
                               "bins": sim_bins},
                "error_by_train_target_quartile": {"train_quartile_edges": [round(float(q), 6) for q in quantiles],
                                                   "bins": quart_stats},
                "prediction_interval": conformal,
                "top_absolute_error_examples": [
                    {"compound_id": val_rows[int(i)]["compound_id"],
                     "csv_row": val_rows[int(i)]["csv_row"],
                     "observed_logd": round(float(yval[i]), 6),
                     "predicted_logd": round(float(predictions[i]), 6),
                     "absolute_error": round(float(abs(predictions[i] - yval[i])), 6),
                     "max_train_tanimoto": round(float(similarity[i]), 6)}
                    for i in worst_idx
                ],
                "validation_predictions_written": True, "test_predictions_written": False,
            }
        (stage / "analysis_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for strategy in SPLIT_STRATEGIES:
            dest = output_dir / strategy
            dest.mkdir(exist_ok=True)
            for path in (stage / strategy).iterdir():
                os.replace(path, dest / path.name)
        os.replace(stage / "analysis_report.json", output_dir / "analysis_report.json")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze molecular generalisation using train/validation only")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--analysis-config", type=Path, default=Path("configs/analysis.json"))
    parser.add_argument("--force", action="store_true", help="Replace earlier analysis artifacts explicitly")
    args = parser.parse_args()
    try:
        result = analyze(args.processed_dir, args.baseline_dir, args.output_dir,
                         args.model_config, args.analysis_config, force=args.force)
    except (AnalysisError, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
