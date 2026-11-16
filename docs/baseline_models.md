# Baseline molecular lipophilicity models

This experiment predicts experimental logD at pH 7.4 using structure-derived molecular features. It compares a random-grouped split and a scaffold-held-out split using the exact assignment files generated during preparation.

## Candidates and feature spaces

| Candidate | Inputs | Settings |
| --- | --- | --- |
| Dummy mean | RDKit descriptors | Predicts the training-set target mean |
| Ridge (descriptors) | 15 RDKit 2D descriptors | Train-only `StandardScaler`; fixed α = 10 |
| Ridge (Morgan) | 2,048 Morgan fingerprint bits | Sparse input; Ridge `lsqr`, fixed α = 10 |
| Random Forest (descriptors) | 15 RDKit 2D descriptors | 150 trees, min leaf size 2, fixed seed 2026 |

Candidates and hyperparameters are **predefined**, not optimised using test labels. These baselines are not intended to constitute a state-of-the-art benchmark. Only training and validation splits are used; `test` is deliberately excluded from model selection and prediction exports.

## Run

From the project directory in Git Bash:

```bash
source .venv/Scripts/activate
python -m pip install -e '.[dev,ml]'
python -m pytest -q -rs
python -m molprop.train
```

A successful run creates `outputs/baselines/training_report.json` and separate subfolders for `random_grouped` and `scaffold_held_out`. Each folder contains four local `joblib` models and a CSV with **validation-only** predictions.

The results report includes train and validation RMSE, MAE, R², the preprocessing configuration checksum, the preparation-report checksum and a validation ranking. No test-set metrics, predictions or model selection are included. Re-running requires an explicit `--force`, because overwriting model artifacts can obscure reproducibility.

## Leakage and interpretation

- The input feature archive and both split files must match the SHA-256 checksums in `preparation_report.json`.
- Validation and training rows must follow the original feature ordering; each split is re-audited for canonical/scaffold overlap before fitting.
- Standardization is performed using `Pipeline(StandardScaler(), Ridge())`, so validation statistics are never used in scaling. Scikit-learn recommends this pattern to prevent preprocessing leakage.
- For each split, model ranking uses validation RMSE only. Differences between random-grouped and scaffold-held-out validation sets may reflect both chemistry differences and target-distribution shifts.
- A scaffold-held-out split does not guarantee low train-validation molecular similarity; a dedicated similarity and uncertainty analysis remains necessary.
- Joblib files must only be loaded from trusted sources (loading untrusted serialized models can execute code).

We will select hyperparameters using development data only. The original 420-molecule test subsets are reserved for a later final evaluation of the selected procedure, after all development decisions are fixed.
