# Publishing this project to GitHub

This document describes how to publish a **new, empty, public** repository named `molecular-property-prediction` under GitHub account `sakthi-kr`. Do not reuse an existing unrelated project repository.

1. In GitHub, choose **New repository**, set the owner to `sakthi-kr` and the name to **molecular-property-prediction**; choose **Public**. Leave **Add README**, **Add .gitignore** and **Choose a license** unchecked, because they are already present in the source.
2. Open Git Bash in your existing `C:\Users\admin\Downloads\molecular-property-prediction` working folder after installing the final source update.
3. Review what is staged before the initial commit:

```bash
cd /c/Users/admin/Downloads/molecular-property-prediction
git init -b main
git add .
python scripts/check_repository.py
git status --short
```

You should see source files, docs, tests and configuration. **Never commit** `data/raw/Lipophilicity.csv`, `.venv`, `data/processed`, `outputs/`, `.joblib`, or real compound-level predictions. `git status --short` should show none of these because `.gitignore` excludes them.

4. Complete the initial push:

```bash
git commit -m "Publish reproducible molecular lipophilicity project"
git remote add origin https://github.com/sakthi-kr/molecular-property-prediction.git
git push -u origin main
```

If the folder is already a Git repository, do not run `git init` or re-add an existing `origin` remote; inspect `git remote -v` and `git status` first. Authenticate with GitHub using Git Credential Manager or your existing Git setup. GitHub account passwords are not accepted for Git over HTTPS.

5. Open **Actions** and check that the **Molecular property prediction tests** workflow passes for Python 3.10 and 3.12. The release workflow is configured to create a one-time `v0.7.0` GitHub Release **after** a successful `main` push CI run. If GitHub Actions is disabled, blocked by repo policy or the workflow has insufficient permissions, do not claim the release exists: inspect the job logs and fix the cause first.

## Repository About metadata

Description: `Molecular lipophilicity prediction with RDKit, scaffold-aware evaluation, frozen ML models and a local Streamlit app.`

Topics: `python`, `machine-learning`, `rdkit`, `chemoinformatics`, `molecular-property-prediction`, `drug-discovery`, `scikit-learn`, `streamlit`, `reproducible-research`

No homepage URL is needed until a real hosted demo exists. The README already includes a CI badge which will show status once GitHub Actions has run.

After publication, verify the new tag at `https://github.com/sakthi-kr/molecular-property-prediction/releases/tag/v0.7.0`. Consider GitHub branch protection and pin ordering separately so the profile remains a curated set of portfolio projects.
