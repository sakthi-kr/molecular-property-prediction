# Experiment protocol

## Research question

How accurately can structure-derived molecular representations predict *experimental* octanol/water distribution coefficient logD at pH 7.4 on held-out molecules?

This is one regression target, **not** binding affinity, drug efficacy, drug safety, or candidate discovery.

## Provenance

Primary benchmark: MoleculeNet Lipophilicity, 4,200 published observations, originally associated with the AstraZeneca / ChEMBL deposited dataset DOI `10.6019/chembl3301361`. Source CSV is fetched from DeepChem's dataset mirror by `molprop.download`. Source checksum is recorded locally. Raw dataset isn't redistributed in the repository, since redistribution terms have not been established.

## Leakage prevention and validity

- Inspect invalid SMILES, missing/invalid targets, canonical duplicates and label conflicts before splitting.
- Do **not** remove suspicious points or average conflicting labels without documenting it.
- Use the fixed random-grouped and Bemis–Murcko scaffold-held-out splits, training / validation / test (target 80/10/10), preserving identical canonical molecules within one partition.
- Train-set-only preprocessing, feature selection and model tuning; do not examine the test metrics during selection.
- Report exact split sizes, overlap audits, target shifts and molecular similarity distributions.
- Scaffold splitting is a useful stress test, *not* a guarantee of prospective generalization.

## Features and splits

The structure-derived 15-feature RDKit descriptor matrix and radius-2, 2048-bit Morgan fingerprints are implemented. Splits are reproducible, grouped, and do not use logD to assign molecules. Full implementation details appear in [preparation.md](preparation.md).

## Models and evaluation design

Four established baselines are implemented: Dummy mean, standardised 2D descriptors + Ridge, radius-2 Morgan fingerprints + Ridge, and descriptor Random Forest. They were trained on train partitions only and assessed on validation data. Training-only grouped cross-validation selected Morgan Ridge `alpha=10` for both strategies. The pre-registered final choice remains the previously fitted Morgan Ridge model, with the other three models reported as comparisons rather than retuned using test results.

Final held-out reporting uses RMSE as the primary metric, plus MAE, R², nearest-training-neighbor Morgan similarity, subgroup errors, a fixed Ridge-versus-Forest cluster bootstrap, and the separate split-conformal Ridge model's empirical interval coverage and width. The primary result is the scaffold-held-out test; the canonical-molecule-grouped random test is secondary. These are distinct datasets, not directly paired replicates.

See [frozen final evaluation protocol](final_evaluation.md). The lock file binds the original prepared data and development reports to pretest decisions. It requires that every fitted model reproduce its saved validation predictions before test labels are accessed. We do not retrain or adjust parameters in the final-evaluation script.

## Deliverables

- Reproducible loader and manifest
- Dataset audit and split audit
- Baseline and training-only CV-selected models
- Frozen held-out benchmark with clear limitations
- Prediction CLI or local demo with explicit applicability limitations
- Tests, CI, diagrams and concise technical README
