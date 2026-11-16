# Reproducibility and local demonstration

## What is included

The public repository includes Python source, fixed experiment configurations, offline tests, CI checks, documentation, and example SMILES. It **does not** include the AstraZeneca-associated Lipophilicity CSV, processed fingerprint/split arrays, compound-level prediction CSVs, `.joblib` files or the developer's local reports. These files are excluded by `.gitignore`.

The clean checkout is fully testable with **no dataset download**:

```bash
python -m pip install -e '.[dev,ml,app]'
python -m pip check
python -m pytest -q -rs
python scripts/check_repository.py
```

## Fresh reproduction of the study

Start in a clean working directory with Python 3.10 or 3.12. Download the source from the cited MoleculeNet endpoint using the supplied script; its SHA-256 manifest is saved locally. Inspect the original data license and reuse terms before publishing or redistributing any data.

```bash
python -m molprop.download
python -m molprop.audit
python -m molprop.prepare
python -m molprop.train
python -m molprop.analyze
```

Training results depend on the feature-generation environment and package builds. The developer's frozen study used **RDKit 2026.09.2**, **scikit-learn 1.7.2**, **NumPy 2.2.6**, and its original preparation reports include the feature-file and split-file SHA-256 hashes. Exact reproduction of the saved pickled models requires those versions and matching configuration/input artifacts; the general CI suite intentionally tests a wider supported package range. `rdkit==2026.9.2` and `scikit-learn==1.7.2` are obtainable from PyPI for supported systems, but a fresh run can still produce different serialized hashes across platforms and dependencies.

The file `configs/final_evaluation.json` is the **historical frozen evaluation contract** and contains checksums from the recorded run. Do **not** edit it to force a different run to look like the original result. A new independent reproduction should be reported as a separate experiment with its own training/configuration checksums and predefined test protocol. This repository reports the original held-out result; no need to rerun that evaluation on the developer's laptop.

## Starting the inference app on the original developer laptop

The previously generated models, prepared fingerprints and final report must remain in their original locations. The application intentionally refuses to load model files when their checksums, scikit-learn version or RDKit fingerprint-generation version disagree with the locked reports. From the project root:

```bash
source .venv/Scripts/activate  # Git Bash on Windows
python -m pip install -e '.[dev,ml,app]'
python -m molprop.predict --smiles 'CCO'
python -m streamlit run src/molprop/app.py
```

Open the local Streamlit address (usually http://localhost:8501), then stop it using **Ctrl+C** in the terminal. The repository does **not** offer a hosted demonstration or pretrained-model download. A new clone without the original artifacts should show a clear missing-artifacts message, not an instant model result.

Batch inference accepts a UTF-8 CSV with `compound_id` and `smiles`, with a maximum of 100 molecules. Example input is provided at [`examples/sample_smiles.csv`](../examples/sample_smiles.csv). Invalid structures are handled per row, without stopping the entire batch. The app is a research demonstration, not a predictor of drug safety, treatment efficacy or clinical utility. All displayed intervals refer to the **separate calibration model** and are not guaranteed for chemically unfamiliar compounds.

## Tests, scope and constraints

The GitHub Actions workflow tests the code with fictional molecules on Python 3.10 and 3.12; it does not download the MoleculeNet dataset or retrain the original published model. This separation is intentional. No GPU, cloud account, API key or paid software is required. Never deserialize `.joblib` files from untrusted sources.
