"""Audit SMILES, logD labels, duplicates and scaffolds before any ML training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


class DataAuditError(ValueError):
    """Dataset missing or structurally incompatible with declared source."""


def _scaffold(mol: Chem.Mol) -> str:
    # Empty scaffold for acyclic molecules must be treated carefully in splitting.
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "<ACYCLIC>"


def audit_dataset(path: Path, report_dir: Path) -> dict:
    """Record all problems; do not modify, deduplicate or split source rows."""
    path = Path(path)
    report_dir = Path(report_dir)
    if not path.is_file():
        raise DataAuditError(f"Dataset not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    valid_records: list[dict] = []
    invalid_records: list[dict] = []
    groups = defaultdict(list)
    scaffold_counts: Counter[str] = Counter()

    try:
        stream = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DataAuditError(f"Cannot read data source: {exc}") from exc
    with stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if not {"smiles", "exp"}.issubset(columns):
            raise DataAuditError(f"Expected columns 'smiles' and 'exp', found: {columns}")
        source_rows = 0
        for row_number, item in enumerate(reader, start=2):
            source_rows += 1
            raw_smiles = (item.get("smiles") or "").strip()
            target_text = (item.get("exp") or "").strip()
            compound = (item.get("CMPD_CHEMBLID") or "").strip()
            reasons = []
            try:
                target = float(target_text)
                if not math.isfinite(target):
                    raise ValueError
            except (TypeError, ValueError):
                target = None
                reasons.append("missing_or_invalid_logd")
            mol = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
            if mol is None or mol.GetNumAtoms() == 0:
                reasons.append("invalid_smiles")
            if reasons:
                invalid_records.append({"csv_row": row_number, "compound_id": compound, "reasons": reasons})
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            scaffold = _scaffold(mol)
            record = {
                "csv_row": row_number, "compound_id": compound,
                "canonical_smiles": canonical, "scaffold": scaffold,
                "logd_7_4": target,
            }
            valid_records.append(record)
            groups[canonical].append(record)
            scaffold_counts[scaffold] += 1
    if not source_rows:
        raise DataAuditError("Dataset has no rows")
    labels = [r["logd_7_4"] for r in valid_records]
    duplicate_groups = {smiles: rows for smiles, rows in groups.items() if len(rows) > 1}
    conflict_groups = {smiles: rows for smiles, rows in duplicate_groups.items()
                       if max(r["logd_7_4"] for r in rows) - min(r["logd_7_4"] for r in rows) > 1e-6}
    # A numeric duplicate may still represent separate measurements. We flag it,
    # rather than arbitrarily deleting or averaging labels.
    report = {
        "dataset": "MoleculeNet Lipophilicity (logD at pH 7.4)",
        "source_file": str(path),
        "source_sha256": digest,
        "rdkit_version": rdBase.rdkitVersion,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": source_rows,
        "expected_original_rows": 4200,
        "row_count_matches_reference": source_rows == 4200,
        "valid_rows": len(valid_records),
        "invalid_rows": len(invalid_records),
        "invalid_examples": invalid_records[:10],
        "unique_canonical_smiles": len(groups),
        "duplicate_canonical_groups": len(duplicate_groups),
        "duplicate_canonical_extra_rows": sum(len(rows) - 1 for rows in duplicate_groups.values()),
        "conflicting_label_groups": len(conflict_groups),
        "conflicting_examples": [
            {"canonical_smiles": smi, "csv_rows": [r["csv_row"] for r in rows],
             "targets": [r["logd_7_4"] for r in rows]}
            for smi, rows in list(conflict_groups.items())[:5]
        ],
        "distinct_scaffolds": len(scaffold_counts),
        "acyclic_rows": scaffold_counts["<ACYCLIC>"],
        "largest_scaffolds": [{"scaffold": group, "rows": count} for group, count in scaffold_counts.most_common(8)],
        "target_summary": {
            "min": min(labels) if labels else None,
            "max": max(labels) if labels else None,
            "mean": round(mean(labels), 5) if labels else None,
            "median": round(median(labels), 5) if labels else None,
        },
        "notes": [
            "No training/validation/test split created. Test data remains untouched.",
            "No rows were changed or dropped in the original source.",
            "Conflicting duplicates require a documented resolution policy before modelling.",
            "Scaffold split alone does not guarantee absence of structural similarity across sets.",
            "Molecular scaffold groups and canonical structure groups must not cross splits.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "valid_records.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["csv_row", "compound_id", "canonical_smiles", "scaffold", "logd_7_4"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(valid_records)
    with (report_dir / "invalid_records.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["csv_row", "compound_id", "reasons"], lineterminator="\n")
        writer.writeheader()
        for item in invalid_records:
            writer.writerow({**item, "reasons": ";".join(item["reasons"])})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit chemical structures, experimental labels and scaffold coverage")
    parser.add_argument("--input", type=Path, default=Path("data/raw/Lipophilicity.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports"))
    args = parser.parse_args()
    try:
        report = audit_dataset(args.input, args.report_dir)
    except (DataAuditError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
