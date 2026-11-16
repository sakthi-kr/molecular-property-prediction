"""Build reproducible molecular representations and locked, leakage-audited splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from rdkit import Chem, rdBase

from molprop.audit import _scaffold
from molprop.representation import DESCRIPTOR_NAMES, calculate_features
from molprop.splitting import SPLITS, SplitError, audit_assignments, make_group_split


class PreparationError(ValueError):
    """Audit provenance, molecular records or preparation configuration is invalid."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Cannot read config: {exc}") from exc
    if cfg.get("descriptor_set") != "rdkit2d-v1":
        raise PreparationError("Unsupported descriptor definition")
    for name in ("random_seed", "morgan_radius", "morgan_bits"):
        if not isinstance(cfg.get(name), int) or isinstance(cfg[name], bool):
            raise PreparationError(f"{name} must be an integer")
    if not isinstance(cfg.get("morgan_include_chirality"), bool):
        raise PreparationError("morgan_include_chirality must be boolean")
    if cfg.get("scaffold_include_chirality") is not False:
        raise PreparationError("Current scaffold groups use non-chiral Bemis-Murcko scaffolds")
    if cfg["morgan_radius"] not in (1, 2, 3) or not 128 <= cfg["morgan_bits"] <= 8192:
        raise PreparationError("Unsupported Morgan fingerprint settings")
    return cfg


def load_audited_records(audit_dir: Path, raw_csv: Path) -> tuple[list[dict], dict]:
    audit_path = audit_dir / "data_audit.json"
    valid_path = audit_dir / "valid_records.csv"
    source_manifest = raw_csv.with_name("source_manifest.json")
    for path in (raw_csv, audit_path, valid_path, source_manifest):
        if not path.is_file():
            raise PreparationError(f"Required local input is missing: {path}. Run download and audit first")
    try:
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        report = json.loads(audit_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise PreparationError(f"Cannot read source manifest or audit: {exc}") from exc
    source_hash = digest(raw_csv)
    if source_hash != manifest.get("sha256") or source_hash != report.get("source_sha256"):
        raise PreparationError("Source CSV changed since download/audit; rerun the original data audit")
    if report.get("invalid_rows", 1) != 0:
        raise PreparationError("Input contains invalid records; document and resolve them first")
    if report.get("conflicting_label_groups", 1):
        raise PreparationError("Conflicting measurements require a documented resolution policy")
    with valid_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"csv_row", "compound_id", "canonical_smiles", "scaffold", "logd_7_4"}:
            raise PreparationError("Invalid audited record columns")
        rows = list(reader)
    if not rows or len(rows) != report.get("valid_rows") or len(rows) != report.get("raw_rows"):
        raise PreparationError("Audited record count inconsistent with source audit")
    seen_rows: set[int] = set()
    for record in rows:
        try:
            row_no = int(record["csv_row"])
            value = float(record["logd_7_4"])
        except (ValueError, TypeError, OverflowError) as exc:
            raise PreparationError("Invalid CSV row ID or logD measurement") from exc
        if row_no in seen_rows or row_no < 2 or not math.isfinite(value):
            raise PreparationError("Repeated row identifier or invalid logD target")
        seen_rows.add(row_no)
        mol = Chem.MolFromSmiles(record["canonical_smiles"])
        if mol is None or mol.GetNumAtoms() == 0:
            raise PreparationError(f"Invalid canonical structure at CSV row {row_no}")
        if Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) != record["canonical_smiles"]:
            raise PreparationError(f"Noncanonical structure at CSV row {row_no}")
        # Reuse the audit's structural metadata; no label-derived features.
        if not record["scaffold"]:
            raise PreparationError(f"Missing scaffold at CSV row {row_no}")
    rows.sort(key=lambda x: int(x["csv_row"]))
    # The audit CSV is generated locally, so verify every derived structure
    # and label against the immutable hashed source, not merely the row count.
    with raw_csv.open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if len(raw_rows) != len(rows):
        raise PreparationError("Raw source row count differs from audited valid records")
    for pos, record in enumerate(rows):
        source = raw_rows[pos]
        if int(record["csv_row"]) != pos + 2 or source.get("CMPD_CHEMBLID", "").strip() != record["compound_id"]:
            raise PreparationError("Audited row identifier does not match the original CSV")
        mol = Chem.MolFromSmiles((source.get("smiles") or "").strip())
        if mol is None or Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) != record["canonical_smiles"]:
            raise PreparationError("Audited canonical structure does not match source CSV")
        if _scaffold(mol) != record["scaffold"] or float(source["exp"]) != float(record["logd_7_4"]):
            raise PreparationError("Audited scaffold or logD differs from source CSV")
    return rows, {"source_sha256": source_hash, "audit_sha256": digest(audit_path),
                  "valid_records_sha256": digest(valid_path)}


def prepare_dataset(
    raw_csv: Path = Path("data/raw/Lipophilicity.csv"),
    audit_dir: Path = Path("data/reports"),
    output_dir: Path = Path("data/processed"),
    config_path: Path = Path("configs/preparation.json"),
    *, force: bool = False,
) -> dict:
    """Prepare features and fixed splits; do not fit models or modify raw data."""
    raw_csv, audit_dir, output_dir, config_path = map(Path, (raw_csv, audit_dir, output_dir, config_path))
    cfg = load_config(config_path)
    rows, source = load_audited_records(audit_dir, raw_csv)
    y = np.asarray([float(r["logd_7_4"]) for r in rows], dtype=np.float64)
    canonical = [r["canonical_smiles"] for r in rows]
    scaffolds = [r["scaffold"] for r in rows]
    per_canonical: dict[str, set[float]] = defaultdict(set)
    for smi, value in zip(canonical, y):
        per_canonical[smi].add(float(value))
    if any(len(targets) > 1 for targets in per_canonical.values()):
        raise PreparationError("Conflicting canonical structures: inspect original audit")
    random_split = make_group_split(canonical, ratios=cfg["split_ratios"], seed=cfg["random_seed"])
    scaffold_split = make_group_split(scaffolds, ratios=cfg["split_ratios"], seed=cfg["random_seed"])
    random_check = audit_assignments(canonical, scaffolds, random_split)
    scaffold_check = audit_assignments(canonical, scaffolds, scaffold_split)
    if scaffold_check["scaffolds_crossing_partitions"] != 0:
        raise PreparationError("Scaffold leakage detected")
    descriptors, fingerprints = calculate_features(canonical, radius=cfg["morgan_radius"],
        bits=cfg["morgan_bits"], include_chirality=cfg["morgan_include_chirality"])
    target_summary = {}
    for key, assignments in (("random_grouped", random_split), ("scaffold_held_out", scaffold_split)):
        # Report target distributions only for the development partitions.
        # Test labels are stored for eventual final evaluation but not
        # summarized or used for split assignment/model selection here.
        target_summary[key] = {
            p: {"count": sum(s == p for s in assignments),
                "logd_mean": round(mean(float(v) for v, s in zip(y, assignments) if s == p), 5),
                "logd_min": float(min(v for v, s in zip(y, assignments) if s == p)),
                "logd_max": float(max(v for v, s in zip(y, assignments) if s == p))}
            for p in ("train", "validation")
        }
    output_files = [output_dir / name for name in
        ("features.npz", "random_grouped.csv", "scaffold_held_out.csv", "preparation_report.json")]
    existing = [str(f) for f in output_files if f.exists()]
    if existing and not force:
        raise PreparationError(f"Outputs already exist; refusing to overwrite: {', '.join(existing)}. Use --force deliberately")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage all outputs together, then replace. Raw data and source audit are never touched.
    with tempfile.TemporaryDirectory(prefix=".molprop-", dir=output_dir) as tmp:
        tempdir = Path(tmp)
        feature_file = tempdir / "features.npz"
        np.savez_compressed(feature_file, descriptors=descriptors, morgan=fingerprints,
                            logd=y, csv_row=np.array([int(r["csv_row"]) for r in rows],dtype=np.int32))
        for file_name, assigned in (("random_grouped.csv", random_split), ("scaffold_held_out.csv", scaffold_split)):
            with (tempdir / file_name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(("csv_row", "compound_id", "canonical_smiles", "scaffold", "split"))
                for row, part in zip(rows, assigned):
                    writer.writerow([row["csv_row"], row["compound_id"], row["canonical_smiles"], row["scaffold"], part])
        report = {
            "dataset": "MoleculeNet Lipophilicity — experimental logD at pH 7.4",
            "rows": len(rows),
            "descriptor_count": len(DESCRIPTOR_NAMES),
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "morgan_fingerprint": {"radius":cfg["morgan_radius"], "bits":cfg["morgan_bits"],
                                   "include_chirality":cfg["morgan_include_chirality"],
                                   "dtype":"uint8"},
            "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__,
            "config_sha256": digest(config_path), **source,
            "data_sha256": {p.name: digest(tempdir / p.name) for p in output_files[:3]},
            "splits": {"random_grouped": random_check, "scaffold_held_out": scaffold_check},
            "target_distributions": target_summary,
            "notes": [
                "No model fitting, feature selection, normalization or hyperparameter tuning occurred.",
                "Features were derived from molecule structures without using logD labels.",
                "Random-grouped forbids identical canonical molecules crossing splits; scaffolds may overlap.",
                "Scaffold-held-out forbids all Bemis-Murcko scaffold groups (including acyclic) crossing splits.",
                "Scaffold separation does not guarantee low Tanimoto similarity; evaluate in a later stage.",
                "Validation and test labels are saved but must not be used for tuning the models.",
                "Outputs are local only; do not commit molecular data or derived structures.",
            ],
        }
        (tempdir / "preparation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for file in output_files:
            os.replace(tempdir / file.name, file)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute molecular representations and leakage-audited train/val/test splits")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/Lipophilicity.csv"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/preparation.json"))
    parser.add_argument("--force", action="store_true", help="Replace generated outputs, never raw input")
    args = parser.parse_args()
    try:
        result = prepare_dataset(args.raw, args.audit_dir, args.output_dir, args.config, force=args.force)
    except (PreparationError, SplitError, OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
