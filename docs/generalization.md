# Molecular generalisation and uncertainty analysis

This analysis uses only the original **train and validation** assignments. Its aim is to test how well structure-based logD models generalise beyond a fixed training set, and to document failure modes before any held-out test evaluation.

## Prerequisites and run

From the project root with the existing virtual environment activated:

```bash
python -m pip install -e '.[dev,ml]'
python -m pytest -q -rs
python -m molprop.analyze
```

The inputs must already exist:

- `data/processed/features.npz`, both split CSV files, and `preparation_report.json` (prepared earlier)
- `outputs/baselines/training_report.json` and each split's `validation_predictions.csv` (produced by `python -m molprop.train`)
- `configs/models.json` and the fixed `configs/analysis.json`

**Do not rerun preparation or baseline training just to run this analysis.** The command verifies source checksums and baseline prediction alignment before computing new results. It does not deserialize the prior `.joblib` models, modify any input, retrain the previous baselines, or evaluate test observations. It will refuse to overwrite a completed `outputs/analysis/analysis_report.json` without `--force`.

## Outputs

`outputs/analysis/analysis_report.json` contains both random-grouped and scaffold-held-out analysis results:

- Training-only **grouped cross-validation** RMSE for Ridge with Morgan fingerprints, using a predefined grid α ∈ {0.1, 1, 10, 100} and three `GroupKFold` folds. Each fold keeps canonical molecules (random strategy) or entire Bemis–Murcko scaffolds (scaffold strategy) together. The alpha with the lowest **mean fold RMSE** is selected, with smaller alpha breaking ties.
- Validation-only RMSE / MAE / R² of the CV-selected model; fixed Ridge Morgan and Random Forest baselines remain as comparisons. Validation observations do not influence the cross-validation hyperparameter selection.
- Per-validation-molecule **maximum Morgan/Tanimoto similarity** to the original training partition. Similarity bins are fixed in `configs/analysis.json`. These are *hashed radius-2 bit-vector similarities*, not proof of equivalent chemical environments. A held-out scaffold can still have a high fingerprint similarity to training molecules.
- Error analysis by fixed similarity bins and **training-target-derived** logD quartile thresholds; worst absolute-error molecules are reported only among validation rows.
- Paired **group-bootstrap** difference in validation RMSE between CV-selected Ridge Morgan and fixed baseline Ridge Morgan, resampling whole scaffold groups for the scaffold split and canonical groups for the random-grouped split. The approximate interval is exploratory: it captures variation in this split, not uncertainty across future chemical campaigns.
- A separate **fixed-alpha split-conformal Ridge Morgan model**: a `GroupShuffleSplit` separates proper-training data and calibration data *within the original training partition*; the model is fitted only to proper-training observations. The nominal 90% interval half-width is the `ceil((n_cal + 1) * 0.90)` order statistic of calibration absolute residuals. Validation is used **only** to report empirical interval coverage and average width. It is not used to calibrate or choose the half-width. **This interval model differs from the full-training CV-selected model.**

For each splitting strategy, the output directory also contains:

- `validation_diagnostics.csv`: the original validation rows, nearest training-neighbor similarity, predictions and interval endpoints. It contains no test rows.
- `cv_selected_ridge_morgan.joblib` and `calibration_ridge_morgan.joblib`: new, local models. Load joblib files **only if they were produced locally or from a trusted source**.
- `observed_vs_predicted.png`, `similarity_vs_error.png`: figures based on validation rows.

**No results from the original 420-molecule test partition are generated or written.** Each strategy has its own independently fitted models and validation data. For the main comparison use scaffold-held-out validation as the more challenging development benchmark, with random-grouped validation reported separately.

## Interpreting the results

Do not choose a final model based solely on the lowest single validation score. Consider the validation RMSE difference, chemical similarity, systematic bias at the target extremes, model robustness, and the interpretation of prediction intervals under distribution shift. The validation data has been consulted for performance analysis; repeatedly trying many further models on this same validation set could still lead to overfitting model selection.

The split-conformal method has marginal coverage guarantees only under the relevant exchangeability assumptions. A group-based calibration split does **not** establish formal 90% coverage on novel held-out scaffolds, nor does it make logD estimates fit for chemical decisions. Report the **empirical** validation coverage and width, and distinguish them from any nominal interval target.

There is no guarantee of clinical efficacy, molecular safety, drug-likeness, binding affinity, experimental reproducibility, or predictive success on prospective chemical series. logD and logP are related but different properties.

## Evaluation protocol and next decision

The final model and calibration procedure have now been frozen against the actual 4,200-record analysis report. See [the held-out final evaluation protocol](final_evaluation.md) for the fixed models, SHA-256 checksums and one-time evaluation guard. Do not change test splits, molecular representations or labels based on test outcomes.
