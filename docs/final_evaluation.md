# Frozen final evaluation of experimental logD

This project uses **one reserved held-out test set per split strategy**, with the modelling decisions recorded before opening either one. The test subsets are not used to choose a model, tune a hyperparameter, set an uncertainty threshold, or change the representation.

## Prespecified decisions

| Setting | Frozen decision |
| --- | --- |
| Primary evaluation | Bemis–Murcko scaffold-held-out test |
| Secondary evaluation | Canonical molecule-grouped random test |
| Main predictor | Previously fitted Morgan fingerprint Ridge regression (`alpha=10`) |
| Comparison predictors | Dummy mean, descriptor Ridge and descriptor Random Forest (already fitted) |
| Calibration model | Separately fitted Ridge on proper training data with training-only conformal residual calibration |
| Prediction intervals | Fixed nominal 90% using the calibration quantile saved during validation analysis |
| Selection metric | Pre-existing validation RMSE; **no test-based model selection** |
| Reported final metrics | RMSE, MAE, R², error-by-similarity, error-by-logD quartile, interval coverage and mean interval width |
| Supplemental comparison | Prespecified cluster bootstrap of Ridge − Random Forest test RMSE |
| Test access | One complete evaluation; no retuning or repeated testing after inspecting results |

These choices are stored in [`configs/final_evaluation.json`](../configs/final_evaluation.json). Its SHA-256 references were frozen against the **actual uploaded preparation, training and analysis reports**. This is a version-specific lock, not a reusable configuration for an independently prepared dataset. The lock binds the earlier dataset/features, both development reports and the model/analysis configurations. It also requires scikit-learn `1.7.2`, the version recorded when the local model files were saved.

## How to run on your Windows laptop

From `C:\Users\admin\Downloads\molecular-property-prediction` in Git Bash:

```bash
source .venv/Scripts/activate
python -m pytest -q -rs
python -m molprop.final_evaluation
```

**Do not** regenerate `data/processed/`, retrain the models or redo Phase 4 beforehand. Doing so changes the artifacts that were frozen before final test access. The command reads existing:

- `data/processed/features.npz`, `data/processed/*.csv` and the preparation report
- `outputs/baselines/training_report.json`, validation CSVs and model `.joblib` files
- `outputs/analysis/analysis_report.json`, validation diagnostics and the saved calibration model
- `configs/models.json`, `configs/analysis.json`, `configs/final_evaluation.json`

The script first checks all report/configuration checksums and both split audits. It verifies that the installed scikit-learn version matches the saved one, then loads **only locally created, trusted joblib models** and confirms each reproduces the saved validation predictions within numerical tolerance. Joblib files use pickle and can execute code; never run this command on models downloaded from an untrusted source. Saved interval predictions and calibration bounds are verified against the previous validation diagnostics. Only after all these checks pass does it access test rows.

If you see a version mismatch, first check the installed version with `python -c "import sklearn; print(sklearn.__version__)"`. For these saved models, install the recorded version with `python -m pip install scikit-learn==1.7.2` if needed; do not disable the version check.

## Outputs

```text
outputs/final_evaluation/
    final_report.json
    random_grouped/
        test_predictions.csv
        test_observed_vs_predicted.png
        test_similarity_vs_error.png
    scaffold_held_out/
        test_predictions.csv
        test_observed_vs_predicted.png
        test_similarity_vs_error.png
```

The report lists model metrics separately for each partition, conditional errors by maximum training-set Morgan similarity and logD quartile, a descriptive group/bootstrap comparison, and coverage/width of the **separate calibration model**. It records SHA-256 checksums of the evaluated model artifacts and test prediction files. The two test partitions should **not** be treated as independent estimates on identical held-out molecules, nor should their scores be mixed into a pooled metric.

The command refuses to run if `outputs/final_evaluation/` already exists. It writes into a temporary folder and publishes a completed directory only after success. This prevents accidentally overwriting the original test report. The results are intended to be final, even if they are disappointing. Do not rerun after inspecting errors in an attempt to optimise the test metric.

## Interpreting the results

The main comparator is the scaffold-held-out Ridge Morgan test RMSE. Random-grouped metrics are a secondary, typically less stringent evaluation. Both use **saved, fitted models**; no `.fit` or cross-validation occurs in the evaluation script.

Prediction intervals are generated with the separate calibration-trained Ridge, not the full-training baseline. They may be wide. Marginal coverage under exchangeability does not guarantee coverage when new scaffolds differ from the calibration distribution.

The benchmark predicts experimental lipophilicity (logD at pH 7.4) for compounds in a curated historic dataset. It does not demonstrate successful drug discovery, biological activity, prospective validation, safety or clinical efficacy. The raw compound data, derived features, model binaries and test-prediction CSVs are excluded from GitHub by `.gitignore` until the data redistribution terms have been checked.

## What happens after this

The application/demo stage must use a frozen model and clearly disclose the evaluation setting and uncertainty limitations. Test-set results should not be used to pick a different architecture. If a subsequent improvement experiment is worthwhile, it needs a separately defined, fresh evaluation protocol and new data.
