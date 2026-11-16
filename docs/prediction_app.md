# Frozen molecular prediction app

The local Streamlit application presents the **frozen, previously evaluated** Morgan fingerprint Ridge model for experimental logD at pH 7.4. It has no training, hyperparameter search or external API calls.

## Prerequisites

Use the same project directory and Python environment from the modelling and final-evaluation runs. Keep the original, locally generated files in place:

- `configs/preparation.json`, `configs/models.json`, `configs/analysis.json`, `configs/final_evaluation.json`
- `data/processed/features.npz`, both split CSVs and `preparation_report.json`
- `outputs/baselines/training_report.json`, validation CSVs and model `.joblib` files for both strategies
- `outputs/analysis/analysis_report.json`, validation diagnostics and the separately fitted calibration model files
- `outputs/final_evaluation/final_report.json`

The app **does not ship model files, raw data or processed molecular records**. The `joblib` format is pickle-based and may execute code on load. Use **only models you trained and saved locally and trust**; never deserialize downloaded or unsolicited joblib files. The loader verifies the frozen report contract, the local model-file hashes, scikit-learn version and RDKit version, and reproduces saved validation predictions before allowing inference.

## Install and start (Windows Git Bash)

```bash
cd /c/Users/admin/Downloads/molecular-property-prediction
source .venv/Scripts/activate
python -m pip install -e '.[dev,ml,app]'
python -m pytest -q -rs
python -m streamlit run src/molprop/app.py
```

Streamlit normally opens `http://localhost:8501`. Stop the server by pressing **Ctrl+C** in the terminal. A clean checkout without local model artifacts will display a missing-input message; it will not retrain or access network services automatically.

You can also obtain a command-line prediction without opening Streamlit:

```bash
python -m molprop.predict --smiles "CCO"
```

### Interpreting a result

1. **Predicted logD** is the output of the full-training Ridge Morgan model selected using validation data. Its final scaffold-held-out benchmark (420 molecules) achieved RMSE 0.923, MAE 0.733 and R² 0.412.
2. **Similarity to training** is the maximum hashed Morgan/Tanimoto similarity between the entered structure and the 3,360 **training** fingerprints in the primary scaffold split. No test fingerprints are included in this similarity reference. It is a *descriptive heuristic*, not a statistically validated applicability-domain probability.
3. **Nominal 90% interval** comes from the **separately fitted proper-training/calibration Ridge model**. Its interval is centered on that separate model's prediction, **not** on the full-training Ridge point estimate. On the scaffold-held-out test, it covered 372 of 420 cases (88.6%) with mean width 3.02 logD units. These intervals can be unreliable under distribution shift.
4. **Descriptors** are RDKit structure-derived 2D features. `MolLogP` is a calculated logP descriptor; it must not be conflated with experimental logD.
5. **Warnings** are displayed for low training similarity, disconnected fragments and charged compounds. These warnings are safeguards, not proof that an unflagged result is accurate.

## Batch CSV use

The CSV batch tab accepts a UTF-8 CSV with `smiles` and optionally `compound_id` columns, up to 100 rows and 200 kB. Each molecule is validated separately; invalid rows are reported rather than causing a silent omission. Downloaded CSVs include primary predictions, separate interval-model outputs, similarity and warnings. Input and predictions remain local to the Streamlit process.

Example:

```csv
compound_id,smiles
example-1,CCO
example-2,c1ccccc1
```

## Safety and scientific limitations

The app does **not** estimate drug efficacy, toxicity, bioavailability, clinical safety, or therapeutic suitability. Its target is experimental logD at pH 7.4. Molecules outside the represented chemical space may have large prediction errors. No retraining or additional held-out testing occurs while using the app. It is not a drug-design decision system.
