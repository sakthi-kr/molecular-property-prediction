# GitHub repository and maintenance

The source is published at [sakthi-kr/molecular-property-prediction](https://github.com/sakthi-kr/molecular-property-prediction). Its first portfolio snapshot, [v0.7.0](https://github.com/sakthi-kr/molecular-property-prediction/releases/tag/v0.7.0), was created from a commit that passed GitHub Actions on Python 3.10 and 3.12.

## What is and is not published

The repository tracks source code, fixed experiment configurations, tests, documentation, and a small example SMILES file. The original MoleculeNet dataset, derived feature arrays, split assignments, model binaries, local reports, and molecular prediction outputs remain excluded. The MIT license applies to repository source code, **not** automatically to the referenced dataset.

The clean clone can run the offline test suite. The Streamlit predictor **cannot** make real predictions until the local, version-compatible trained model artifacts and their verification metadata are available. See [reproducibility](reproducibility.md) and [prediction app](prediction_app.md).

## Updating the existing repository

From the working Git Bash project folder:

```bash
git status --short
python scripts/check_repository.py
python -m pytest -q -rs
git add .
python scripts/check_repository.py
git diff --cached --stat
```

Review the staged files before committing. Do not add downloaded molecular records, model `.joblib` files, processed arrays, held-out compound predictions, personal secrets, or large generated outputs.

For nontrivial changes, create a feature branch, open a pull request and confirm both CI jobs pass before merging. Preserve the frozen experimental contracts and the already evaluated test scores; any new experiment requires a separate, preregistered evaluation plan rather than modifying the existing historical report.

## Repository metadata

- Description: `Molecular lipophilicity prediction with RDKit, scaffold-aware evaluation, frozen ML models and a local Streamlit app.`
- Topics: `python`, `machine-learning`, `rdkit`, `chemoinformatics`, `molecular-property-prediction`, `drug-discovery`, `scikit-learn`, `streamlit`, `reproducible-research`
- Homepage: leave empty unless a public, tested application is actually hosted.

These settings are configured through the **About** gear icon on GitHub. Profile README inclusion and pin placement are separate editorial decisions.

## CI, release and repository controls

The [Molecular property prediction tests](../.github/workflows/tests.yml) workflow runs unit tests, dependency checks and tracked-file hygiene on Python 3.10 and 3.12. The [release workflow](../.github/workflows/release.yml) published `v0.7.0` only once after the main-branch CI run succeeded.

For ongoing maintenance, consider **Settings → Rules → Rulesets** or the branch-protection settings available to your account. Require passing CI and pull requests for `main` once those requirements fit your single-maintainer workflow; verify the rules are actually enforced rather than merely configured. Do not rewrite or retag the published `v0.7.0` release for subsequent documentation changes.
