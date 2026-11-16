"""Frozen, local molecular logD inference from a single SMILES structure.

No training, tuning, test-target selection, network calls, or file writes occur.
Only load locally created and trusted joblib model files: joblib uses pickle.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix
from sklearn import __version__ as sklearn_version

from molprop.analyze import _bits_to_rdkit
from molprop.final_evaluation import (
    FinalEvaluationError,
    _preflight,
    load_lock,
)
from molprop.prepare import digest, load_config
from molprop.representation import DESCRIPTOR_NAMES, calculate_features
from molprop.train import SPLIT_STRATEGIES


class PredictionError(ValueError):
    """Input molecule or frozen inference bundle is invalid."""


@dataclass(frozen=True)
class FrozenBundle:
    """Locally verified frozen models and *training-only* similarity reference."""

    full_model: Any
    calibration_model: Any
    calibration_quantile: float
    nominal_coverage: float
    test_coverage: float
    test_rmse: float
    test_mae: float
    test_r2: float
    train_fingerprints: tuple[Any, ...]
    train_identifiers: tuple[str, ...]
    train_smiles: tuple[str, ...]
    radius: int
    bits: int
    include_chirality: bool
    strategy: str
    sklearn_version: str


def _validate_final_report(report: dict[str, Any], lock: dict[str, Any], *, lock_path: Path) -> dict[str, Any]:
    """Check immutable study and model-selection contract before deserialization."""
    for key, value in (
        ("dataset", lock["dataset"]),
        ("protocol_lock_sha256", digest(lock_path)),
        ("preparation_report_sha256", lock["preparation_report_sha256"]),
        ("training_report_sha256", lock["training_report_sha256"]),
        ("analysis_report_sha256", lock["analysis_report_sha256"]),
        ("primary_split", lock["primary_split"]),
        ("primary_model", lock["primary_model"]),
        ("sklearn_version", lock["expected_sklearn_version"]),
    ):
        if report.get(key) != value:
            raise PredictionError(f"Final evaluation report disagrees with frozen protocol: {key}")
    if set(report.get("strategies", {})) != set(SPLIT_STRATEGIES):
        raise PredictionError("Final report must contain both frozen evaluation strategies")
    primary = report["strategies"][lock["primary_split"]]
    if primary.get("split_role") != "primary" or primary.get("test_rows") != lock["expected_test_rows_per_split"]:
        raise PredictionError("Primary test-report metadata is inconsistent")
    if not (0 <= float(primary["prediction_interval"]["empirical_test_coverage"]) <= 1):
        raise PredictionError("Invalid reported interval coverage")
    if float(primary["prediction_interval"]["nominal_coverage"]) != lock["nominal_interval_coverage"]:
        raise PredictionError("Final interval coverage target changed")
    return primary


def _verify_saved_models(report: dict[str, Any], baseline_dir: Path, analysis_dir: Path) -> None:
    """Check all files prior to the existing preflight's joblib deserialisation."""
    for strategy, details in report["strategies"].items():
        manifest = details.get("model_artifact_sha256")
        expected = {
            "dummy_mean", "ridge_descriptors", "ridge_morgan",
            "random_forest_descriptors", "calibration_ridge_morgan",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected:
            raise PredictionError(f"Missing frozen model checksums for {strategy}")
        for name, sha in manifest.items():
            path = ((analysis_dir if name == "calibration_ridge_morgan" else baseline_dir)
                    / strategy / f"{name}.joblib")
            if not path.is_file() or digest(path) != sha:
                raise PredictionError(f"Frozen model checksum mismatch: {path}")


def load_frozen_bundle(
    *,
    processed_dir: Path = Path("data/processed"),
    baseline_dir: Path = Path("outputs/baselines"),
    analysis_dir: Path = Path("outputs/analysis"),
    final_report_path: Path = Path("outputs/final_evaluation/final_report.json"),
    lock_path: Path = Path("configs/final_evaluation.json"),
    model_config: Path = Path("configs/models.json"),
    analysis_config: Path = Path("configs/analysis.json"),
    preparation_config: Path = Path("configs/preparation.json"),
) -> FrozenBundle:
    """Verify source and previous validation results, then load trusted models.

    The existing provenance checker reads the prepared archive, which contains
    all original labels, but selects *only validation labels* for replay checks.
    No held-out test labels are selected or used in inference/model selection.
    The published final report supplies aggregate test metrics only.
    """
    processed_dir, baseline_dir, analysis_dir, final_report_path = map(
        Path, (processed_dir, baseline_dir, analysis_dir, final_report_path)
    )
    lock_path, model_config, analysis_config, preparation_config = map(
        Path, (lock_path, model_config, analysis_config, preparation_config)
    )
    try:
        lock = load_lock(lock_path)
        final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
        if not isinstance(final_report, dict):
            raise PredictionError("Final evaluation report must be a JSON object")
        primary = _validate_final_report(final_report, lock, lock_path=lock_path)
        if sklearn_version != lock["expected_sklearn_version"]:
            raise PredictionError(
                f"Frozen models require scikit-learn {lock['expected_sklearn_version']}; "
                f"installed version: {sklearn_version}"
            )
        _verify_saved_models(final_report, baseline_dir, analysis_dir)
        # Existing preflight replays saved VALIDATION predictions to verify the
        # model fits; it does not index the TEST partition or re-fit models.
        verified_lock, arrays, assignments, training, analysis, models = _preflight(
            processed_dir, baseline_dir, analysis_dir,
            model_config, analysis_config, lock_path,
        )
        if verified_lock != lock:
            raise PredictionError("Preflight protocol differs from frozen report")
        prep = load_config(preparation_config)
        source = json.loads((processed_dir / "preparation_report.json").read_text(encoding="utf-8"))
        if digest(preparation_config) != source["config_sha256"]:
            raise PredictionError("Feature configuration does not match original preparation")
        if rdBase.rdkitVersion != source["rdkit_version"]:
            raise PredictionError(
                f"Original fingerprints used RDKit {source['rdkit_version']}, "
                f"but this environment has {rdBase.rdkitVersion}. Use the original RDKit version."
            )
        fp_cfg = source["morgan_fingerprint"]
        if (fp_cfg["radius"] != prep["morgan_radius"]
                or fp_cfg["bits"] != prep["morgan_bits"]
                or fp_cfg["include_chirality"] != prep["morgan_include_chirality"]):
            raise PredictionError("Fingerprint settings disagree with original feature generation")
        strategy = lock["primary_split"]
        rows = assignments[strategy]
        train_idx = np.array([i for i, record in enumerate(rows) if record["split"] == "train"], dtype=np.int64)
        if len(train_idx) == 0 or len(set(rows[i]["canonical_smiles"] for i in train_idx)) != len(train_idx):
            raise PredictionError("Missing or duplicated training molecular structures")
        train_bits = arrays["morgan"][train_idx]
        training_fingerprints = tuple(_bits_to_rdkit(train_bits))
        if not all(len(fp) == prep["morgan_bits"] for fp in training_fingerprints):
            raise PredictionError("Training fingerprint width mismatch")
        q = float(analysis["strategies"][strategy]["prediction_interval"]["quantile_logd"])
        if (not math.isfinite(q) or q <= 0
                or abs(q - float(primary["prediction_interval"]["calibration_quantile_logd"])) > 1e-6):
            raise PredictionError("Saved calibration residual quantile disagrees with final report")
        metrics = primary["model_metrics"][lock["primary_model"]]
        for key in ("rmse", "mae", "r2"):
            if not math.isfinite(float(metrics[key])):
                raise PredictionError("Invalid frozen performance metric")
        return FrozenBundle(
            full_model=models[strategy]["ridge_morgan"],
            calibration_model=models[strategy]["calibration_ridge_morgan"],
            calibration_quantile=q,
            nominal_coverage=float(lock["nominal_interval_coverage"]),
            test_coverage=float(primary["prediction_interval"]["empirical_test_coverage"]),
            test_rmse=float(metrics["rmse"]), test_mae=float(metrics["mae"]),
            test_r2=float(metrics["r2"]),
            train_fingerprints=training_fingerprints,
            train_identifiers=tuple(rows[int(i)]["compound_id"] for i in train_idx),
            train_smiles=tuple(rows[int(i)]["canonical_smiles"] for i in train_idx),
            radius=int(fp_cfg["radius"]), bits=int(fp_cfg["bits"]),
            include_chirality=bool(fp_cfg["include_chirality"]),
            strategy=strategy, sklearn_version=sklearn_version,
        )
    except (FinalEvaluationError, OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, PredictionError):
            raise
        raise PredictionError(f"Frozen inference inputs could not be verified: {exc}") from exc


def canonicalize_input(smiles: str) -> tuple[str, Chem.Mol]:
    """Reject invalid/large structures; use precisely the preparation's canonicalization."""
    if not isinstance(smiles, str):
        raise PredictionError("SMILES input must be text")
    text = smiles.strip()
    if not text or len(text) > 1000 or "\n" in text or "\r" in text or "\t" in text:
        raise PredictionError("Enter one valid SMILES string (maximum 1000 characters)")
    mol = Chem.MolFromSmiles(text)
    if mol is None or mol.GetNumAtoms() == 0:
        raise PredictionError("RDKit could not parse this molecular structure")
    if mol.GetNumAtoms() > 150:
        raise PredictionError("Molecule is too large for this local demonstration (over 150 atoms)")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol


def _descriptive_similarity_band(score: float) -> tuple[str, str]:
    """Heuristic display labels, not calibrated applicability-domain classification."""
    if score < 0.3:
        return "Very low similarity", "The compound is highly dissimilar to the training molecules. Treat this estimate with particular caution."
    if score < 0.5:
        return "Low similarity", "The compound is distant from its nearest training neighbour; errors were larger in these regions on the held-out test."
    if score < 0.7:
        return "Moderate similarity", "The compound has some structural overlap with the training set, but reliability is not guaranteed."
    return "Higher similarity", "The compound has a relatively similar training neighbour, but this does not establish prediction accuracy."


def predict_smiles(bundle: FrozenBundle, smiles: str) -> dict[str, Any]:
    """Predict structure-only logD and training-set similarity. Does not write files."""
    canonical, mol = canonicalize_input(smiles)
    desc, bits = calculate_features(
        [canonical], radius=bundle.radius,
        bits=bundle.bits, include_chirality=bundle.include_chirality,
    )
    sample = csr_matrix(bits, dtype=np.float64)
    try:
        point = float(np.asarray(bundle.full_model.predict(sample)).reshape(-1)[0])
        separate_center = float(np.asarray(bundle.calibration_model.predict(sample)).reshape(-1)[0])
    except Exception as exc:
        raise PredictionError(f"Frozen model could not make a prediction: {exc}") from exc
    if not (math.isfinite(point) and math.isfinite(separate_center)):
        raise PredictionError("Nonfinite prediction")
    query_fp = rdFingerprintGenerator.GetMorganGenerator(
        radius=bundle.radius, fpSize=bundle.bits, includeChirality=bundle.include_chirality
    ).GetFingerprint(mol)
    scores = DataStructs.BulkTanimotoSimilarity(query_fp, list(bundle.train_fingerprints))
    if not scores or not all(math.isfinite(float(s)) and 0 <= s <= 1 for s in scores):
        raise PredictionError("Cannot calculate training-neighbour similarity")
    best = int(np.argmax(scores))
    similarity = float(scores[best])
    label, interpretation = _descriptive_similarity_band(similarity)
    warnings = []
    if len(Chem.GetMolFrags(mol)) > 1:
        warnings.append("Multiple disconnected molecular fragments: predictions on mixtures or salts may be unreliable.")
    if Chem.GetFormalCharge(mol) != 0:
        warnings.append("Charged structure: logD depends on ionisation and experimental conditions; this model may be unreliable.")
    if similarity < 0.5:
        warnings.append("Low similarity to the training set; substantial prediction errors were observed for unfamiliar chemistry.")
    warnings.append("Prediction intervals use a DIFFERENT calibration model; they are not centred on the full-training prediction.")
    return {
        "input_smiles": smiles.strip(),
        "canonical_smiles": canonical,
        "predicted_logd": round(point, 6),
        "calibration_model_prediction_logd": round(separate_center, 6),
        "prediction_interval_lower": round(separate_center - bundle.calibration_quantile, 6),
        "prediction_interval_upper": round(separate_center + bundle.calibration_quantile, 6),
        "nominal_interval_coverage": bundle.nominal_coverage,
        "scaffold_test_interval_coverage": bundle.test_coverage,
        "max_train_tanimoto": round(similarity, 6),
        "similarity_band": label,
        "similarity_explanation": interpretation,
        "nearest_train_compound_id": bundle.train_identifiers[best],
        "nearest_train_canonical_smiles": bundle.train_smiles[best],
        "molecular_descriptors": {name: round(float(desc[0, i]), 6) for i, name in enumerate(DESCRIPTOR_NAMES)},
        "warnings": warnings,
    }



# Batch inference follows the same immutable single-molecule contract.
BATCH_COLUMNS = (
    "compound_id", "smiles", "status", "predicted_logd", "interval_model_prediction_logd",
    "interval_lower", "interval_upper", "max_train_tanimoto", "nearest_train_compound_id",
    "similarity_band", "cautions",
)


def decode_smiles_csv(payload: bytes) -> list[tuple[str, str]]:
    """Parse a bounded local CSV input. No network, models, or label fields."""
    if len(payload) > 200_000:
        raise PredictionError("CSV exceeds the 200 kB local demo limit")
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), strict=True)
        if not reader.fieldnames or "smiles" not in reader.fieldnames:
            raise PredictionError("CSV must contain a smiles column; compound_id is optional")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise PredictionError("CSV has duplicate column names")
        rows = []
        for ix, row in enumerate(reader):
            if ix >= 100:
                raise PredictionError("Maximum 100 molecules per batch")
            if None in row:
                raise PredictionError("CSV row contains more fields than the header")
            rows.append((str(row.get("compound_id") or f"row-{ix + 1}"), str(row.get("smiles") or "")))
        if not rows:
            raise PredictionError("No SMILES rows found")
        return rows
    except (UnicodeError, csv.Error) as exc:
        raise PredictionError(f"Invalid UTF-8 CSV: {exc}") from exc


def predict_batch_rows(bundle: FrozenBundle, molecules: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Return row-specific error statuses instead of abandoning a full batch."""
    if not 1 <= len(molecules) <= 100:
        raise PredictionError("Batch must contain 1 to 100 molecules")
    rows: list[dict[str, Any]] = []
    for compound_id, smiles in molecules:
        try:
            out = predict_smiles(bundle, smiles)
            rows.append({
                "compound_id": compound_id, "smiles": smiles, "status": "OK",
                "predicted_logd": out["predicted_logd"],
                "interval_model_prediction_logd": out["calibration_model_prediction_logd"],
                "interval_lower": out["prediction_interval_lower"],
                "interval_upper": out["prediction_interval_upper"],
                "max_train_tanimoto": out["max_train_tanimoto"],
                "nearest_train_compound_id": out["nearest_train_compound_id"],
                "similarity_band": out["similarity_band"],
                "cautions": " | ".join(out["warnings"]),
            })
        except PredictionError as exc:
            rows.append({"compound_id": compound_id, "smiles": smiles, "status": f"INVALID: {exc}"})
    return rows


def _safe_csv_text(value: Any) -> Any:
    """Avoid formula interpretation of untrusted compound identifiers in Excel."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def prediction_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    file = io.StringIO(newline="")
    writer = csv.DictWriter(file, fieldnames=BATCH_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _safe_csv_text(row.get(field, "")) for field in BATCH_COLUMNS})
    return file.getvalue().encode("utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local frozen-model prediction of molecular logD at pH 7.4")
    parser.add_argument("--smiles", required=True, help="A single SMILES string")
    args = parser.parse_args()
    try:
        result = predict_smiles(load_frozen_bundle(), args.smiles)
    except PredictionError as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    # Never print training or test labels; report only inference and provenance metadata.
    print(json.dumps({k: v for k, v in result.items() if k != "molecular_descriptors"}, indent=2))


if __name__ == "__main__":
    main()
