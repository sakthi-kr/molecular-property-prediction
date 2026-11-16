# Molecular Property Prediction — Lipophilicity (logD)

[![Molecular property prediction tests](https://github.com/sakthi-kr/molecular-property-prediction/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/sakthi-kr/molecular-property-prediction/actions/workflows/tests.yml)

Reproducible scientific machine learning for predicting experimental molecular lipophilicity from chemical structure. The target is octanol/water distribution coefficient **logD at pH 7.4** — not logP, toxicity, target binding or drug efficacy.

**Status:** Scientific evaluation complete; [v0.7.0](https://github.com/sakthi-kr/molecular-property-prediction/releases/tag/v0.7.0) published with passing Python 3.10 and 3.12 CI. The prediction interface requires locally generated, checksum-verified artifacts; no pretrained models or hosted API are supplied.

## Results at a glance

| Measure | Frozen result |
| --- | --- |
| Dataset | 4,200 published structures; zero invalid SMILES or duplicate canonical structures in the audited source |
| Molecular representations | 15 RDKit descriptors, 2,048-bit Morgan fingerprints |
| Splitting | Random-grouped and scaffold-held-out 80/10/10 partitions with leakage checks |
| Primary model | Ridge on Morgan fingerprints (`alpha=10`) selected using validation data |
| Scaffold-held-out test | **RMSE 0.923**, **MAE 0.733**, **R² 0.412** across 420 molecules |
| Secondary random-grouped test | Ridge Morgan RMSE **0.792** across 420 molecules |
| Interval evaluation | Separate calibration model; nominal 90% interval covered 88.6% of scaffold-test molecules; mean width 3.02 logD |

**Limitations:** Scores are retrospective on one benchmark. Errors increase for chemically unfamiliar molecules, the prediction intervals are broad, and no drug discovery or clinical performance is established. See [complete results](docs/results.md) and [reproducibility](docs/reproducibility.md).

## Dataset

MoleculeNet Lipophilicity: 4,200 published molecular measurements associated with AstraZeneca / ChEMBL (DOI `10.6019/chembl3301361`). Raw source structures are SMILES, and experimental logD values are in the `exp` column. The data is downloaded locally; raw and processed molecular records are excluded from GitHub until reuse and redistribution terms are established.

## Quick start (Windows Git Bash)

Use one local project directory (for example `molecular-property-prediction`). A new checkout supports tests immediately after dependency installation, but **not predictions** until the dataset and compatible local model artifacts have been generated. See [reproducibility](docs/reproducibility.md).

```bash
# Create and activate a project environment if starting fresh
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash

# Install project tools and tests
python -m pip install -e '.[dev,ml]'
python -m pytest -q -rs
python scripts/check_repository.py

# These two are required only once, if the files do not already exist:
# python -m molprop.download
# python -m molprop.audit

# Generate structure features and locked splits (if not already run)
# python -m molprop.prepare

# Run this only if outputs/baselines/ has not yet been created:
# python -m molprop.train

# Run validation-only analysis if it has not yet been created:
# python -m molprop.analyze

# The frozen held-out evaluation has already been completed: DO NOT rerun it.
# Start the local inference demo after installing optional interface tools:
python -m pip install -e '.[dev,ml,app]'
python -m streamlit run src/molprop/app.py

# Or make one command-line prediction:
# python -m molprop.predict --smiles "CCO"
```

The project requires Python 3.10 or newer (up to, but excluding, 3.15). No GPU, cloud service or paid software is needed for the current workflow.

## What the pipeline does

1. Downloads and SHA-256 records the published molecular data without committing it.
2. Audits SMILES, finite experimental measurements, canonical duplicates and Bemis–Murcko scaffolds.
3. Rechecks the source checksum and audited rows before feature generation.
4. Computes **15 RDKit 2D descriptors** and **2,048-bit Morgan fingerprints (radius 2)** from structures alone.
5. Creates two seeded data partitions: **canonical-molecule-grouped random** and **scaffold-held-out** train/validation/test splits (80/10/10 target proportions).
6. Audits canonical-group overlap and scaffold overlap, records actual split sizes and writes a preparation report.
7. Trains four fixed baselines on training data, compares RMSE/MAE/R² on validation data and preserves the original test partitions for final evaluation.
8. Audits validation errors versus molecular similarity and logD ranges, selects a Ridge-Morgan regularisation value by train-only group cross-validation, compares validation metrics, and assesses a separately calibrated prediction interval.
9. Performs a frozen, one-time held-out evaluation of the saved models, with preflight provenance and validation-prediction checks. Reports metrics for scaffold-held-out and random-grouped tests separately without retraining.
10. Loads locally fitted frozen models in a Streamlit/CLI inference app, displaying the primary logD estimate, training-only molecular similarity, and **separate-model** prediction intervals with uncertainty cautions.

## Frozen held-out results

The selected **Ridge Morgan** model (`alpha=10`) achieved **RMSE 0.923, MAE 0.733 and R² 0.412** on 420 scaffold-held-out test molecules. On the same partition, a Random Forest baseline had RMSE 0.928; the difference was small. The separately calibrated model's nominal 90% intervals covered **88.6%** of those test measurements, with mean interval width **3.02 logD units**. Low-similarity compounds showed worse errors. These are retrospective results on one published dataset, not evidence of prospective drug-discovery performance.

The raw source is never modified, and split assignment never uses property labels. Test data is accessed only after model choices are frozen and validated against earlier artifacts. A scaffold-held-out split is a useful stress test, not a guarantee of novelty or prospective generalisation.

## Open-source source tree and local-only artifacts

This repository includes *only code and metadata*. The raw 4,200-row CSV, molecular fingerprints, split assignment tables, saved `.joblib` models, and individual predictions are **not committed**. A clean clone can validate the source and run tests, but the prediction app requires regenerated artifacts with the original frozen dependencies and checksums. Do not deserialize untrusted pickle/joblib files. An [example SMILES CSV](examples/sample_smiles.csv) provides demo inputs without bundling published molecular data.

## Files

| Path | Purpose |
| --- | --- |
| `configs/preparation.json` | Fixed seed, group split ratios and fingerprint definitions |
| `configs/models.json` | Fixed baseline model settings |
| `configs/analysis.json` | Group CV, similarity, bootstrap and calibration settings |
| `configs/final_evaluation.json` | Frozen model selection, data/report checksums and one-time held-out test contract |
| `src/molprop/` | Downloader, dataset audit, molecular representations, splitting and preparation |
| `tests/` | Offline unit and integration checks using fictional molecules |
| `docs/` | Modelling protocol and split/feature design |
| `data/raw/` | Local-only original CSV and source hash manifest |
| `data/reports/` | Local-only data audit reports |
| `data/processed/` | Local-only features, two split CSV files and preparation report |
| `outputs/baselines/` | Local-only fitted models, validation predictions and metrics |
| `outputs/analysis/` | Local-only validation diagnostics, similarity figures, CV selection and interval results |
| `outputs/final_evaluation/` | Local-only final test metrics, predictions and plots (already generated; do not rerun) |
| `src/molprop/predict.py`, `src/molprop/app.py` | Verified frozen-model inference backend and local Streamlit interface |

## Documentation and limitations

- [Experimental results and limitations](docs/results.md)
- [Reproducibility and local model artifacts](docs/reproducibility.md)
- [Repository maintenance and publication details](docs/github_setup.md)
- [Changelog](CHANGELOG.md)
- [Experiment protocol](docs/experiment_protocol.md)
- [Molecular features and leakage-audited splits](docs/preparation.md)
- [Baseline models and evaluation protocol](docs/baseline_models.md)
- [Generalisation, error analysis and prediction intervals](docs/generalization.md)
- [Frozen final held-out evaluation protocol](docs/final_evaluation.md)
- [Local molecular prediction app and scientific limitations](docs/prediction_app.md)
- [MoleculeNet Lipophilicity dataset reference](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html)
- [Original AstraZeneca/ChEMBL deposition](https://doi.org/10.6019/chembl3301361)
- [RDKit](https://www.rdkit.org/docs/Install.html)

The project does not predict safety, clinical effectiveness or suitability for human use. No patient data, molecular synthesis or therapeutic recommendations are involved.

## License and release

The first public portfolio snapshot, [v0.7.0](https://github.com/sakthi-kr/molecular-property-prediction/releases/tag/v0.7.0), was published after passing CI. The source **code** is MIT-licensed (see [LICENSE](LICENSE)); this does not extend any license to the downloaded molecular data.
