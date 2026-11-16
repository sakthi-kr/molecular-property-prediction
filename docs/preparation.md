# Molecular representations and fixed splits

This stage produces model inputs and **fixed evaluation partitions**, not trained models or performance claims. The measurements are experimental logD at pH 7.4. RDKit's *calculated* MolLogP is one predictor among the 15 two-dimensional descriptors; it is **not** the experimental logD target.

## Inputs and provenance

The preparation command requires three files already created by the download and audit commands:

- `data/raw/Lipophilicity.csv` and `data/raw/source_manifest.json`
- `data/reports/data_audit.json`
- `data/reports/valid_records.csv`

Before processing, the program checks the raw CSV's SHA-256 against both the original download manifest and audit report, then re-checks every audited canonical SMILES, scaffold and logD label against the raw source. It refuses invalid rows and conflicting canonical measurements rather than making an undocumented cleaning decision. The original input files are not modified.

## Features

`features.npz` contains arrays indexed in the same order as the split CSV files:

- `descriptors`: 15 numerical RDKit 2D descriptors, float64, with names documented in `preparation_report.json`. These include molecular weight, RDKit calculated logP, TPSA, H-bond donor/acceptor counts, ring counts, rotatable bonds and physicochemical descriptors.
- `morgan`: 2,048-bit binary Morgan fingerprints, radius 2, including chirality, uint8. Created using RDKit's fingerprint generator, not fitted to the data.
- `logd`: experimental logD targets, float64, retained for controlled model evaluation.
- `csv_row`: original source row identifiers, int32, for traceability.

No descriptor scaling, normalization, imputation, feature selection, model fitting or hyperparameter tuning happens during this stage. All future learned preprocessing must be fitted **only on training molecules**.

## Dataset partitions

Two independent, seeded 80/10/10 train/validation/test assignments are written:

| File | Group unit | Permits shared scaffolds? |
| --- | --- | --- |
| `random_grouped.csv` | Canonical isomeric SMILES | Yes, but not the exact same canonical molecule |
| `scaffold_held_out.csv` | Non-chiral Bemis–Murcko scaffold | **No**; even acyclic molecules share one `<ACYCLIC>` group |

Each source row appears exactly once in each file. The algorithm sorts groups by size and uses a seeded shuffle to break equal-size ties before balancing groups among partitions. This makes the split reproducible without using property labels to choose groups. Exact 80/10/10 row counts may be impossible when large groups are indivisible, so actual sizes and deviations are reported.

**Critical:** Scaffold separation does not guarantee low similarity of molecules across partitions. Tanimoto similarity, nearest-training-neighbor diagnostics and out-of-domain error analysis are reserved for later modelling and evaluation. Do not change or regenerate the splits in response to a disappointing result. Do not use the locked test labels for model selection.

## Usage

From the single project root on Windows Git Bash:

```bash
python -m pip install -e '.[dev,ml]'
python -m pytest -q -rs
python -m molprop.prepare
```

The command refuses to overwrite existing outputs; use `python -m molprop.prepare --force` **only when intentionally regenerating them**. Changed splits after modelling invalidate comparisons, and their changes must be documented. The raw dataset is never overwritten by this command.

Outputs under `data/processed/`:

- `features.npz`
- `random_grouped.csv`
- `scaffold_held_out.csv`
- `preparation_report.json`

They remain local and are excluded from GitHub, since source redistribution terms have not been established. Tests use tiny fictional molecular inputs and do not require network access.
