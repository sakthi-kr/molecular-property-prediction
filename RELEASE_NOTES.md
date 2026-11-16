# Molecular Property Prediction — v0.7.0

Reproducible scientific ML demonstration for predicting experimental molecular lipophilicity (logD at pH 7.4) from SMILES.

- MoleculeNet Lipophilicity (4,200 molecules), with local download, checksum, SMILES/duplicate/scaffold audits.
- 15 RDKit descriptors and 2,048-bit Morgan fingerprints; random-grouped and scaffold-held-out splits.
- Dummy, Ridge and Random Forest baseline comparisons, group-aware validation, error analysis and separately calibrated intervals.
- Frozen scaffold-held-out Ridge Morgan test RMSE **0.923**, MAE **0.733**, R² **0.412**, from 420 molecules. The separately calibrated interval model covered **88.6%** of test measurements at nominal 90% coverage with mean width **3.02 logD units**.
- Local Streamlit/CLI demonstration with molecular rendering, training-set similarity, clear caution messages and bounded CSV inference.
- Offline tests and GitHub Actions CI. Raw data, molecular records, fitted models and prediction CSVs are **not redistributed**.

The source code is MIT-licensed; this does not grant rights over the original dataset. All results are retrospective benchmark scores, not evidence of prospective drug discovery, clinical efficacy or safety. The inference app requires locally generated, version-compatible model artifacts; this release does not provide a hosted predictor.
