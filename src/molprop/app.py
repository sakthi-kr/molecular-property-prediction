"""Local Streamlit demonstration of frozen molecular logD inference.

Run from project root:
    python -m streamlit run src/molprop/app.py

All predictions stay on the local machine. No model training or external API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from molprop.predict import (
    PredictionError, decode_smiles_csv, load_frozen_bundle,
    predict_batch_rows, predict_smiles, prediction_csv_bytes,
)


st.set_page_config(
    page_title="Molecular Lipophilicity | logD Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT = Path(__file__).resolve().parents[2]


@st.cache_resource(show_spinner="Verifying frozen molecular models…")
def _load_models(project: str, signature: tuple[tuple[str, int, int], ...]):
    # The signature invalidates the cache if a local artifact changes; the
    # backend separately verifies every relevant checksum before loading.
    root = Path(project)
    return load_frozen_bundle(
        processed_dir=root / "data/processed",
        baseline_dir=root / "outputs/baselines",
        analysis_dir=root / "outputs/analysis",
        final_report_path=root / "outputs/final_evaluation/final_report.json",
        lock_path=root / "configs/final_evaluation.json",
        model_config=root / "configs/models.json",
        analysis_config=root / "configs/analysis.json",
        preparation_config=root / "configs/preparation.json",
    )


def _artifact_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    files = [
        "configs/final_evaluation.json",
        "configs/preparation.json",
        "configs/models.json",
        "configs/analysis.json",
        "data/processed/features.npz",
        "data/processed/preparation_report.json",
        "data/processed/scaffold_held_out.csv",
        "data/processed/random_grouped.csv",
        "outputs/baselines/training_report.json",
        "outputs/analysis/analysis_report.json",
        "outputs/final_evaluation/final_report.json",
    ]
    for split in ("scaffold_held_out", "random_grouped"):
        files.extend([
            f"outputs/baselines/{split}/{model}.joblib"
            for model in ("dummy_mean", "ridge_descriptors", "ridge_morgan", "random_forest_descriptors")
        ])
        files.extend((
            f"outputs/baselines/{split}/validation_predictions.csv",
            f"outputs/analysis/{split}/calibration_ridge_morgan.joblib",
            f"outputs/analysis/{split}/validation_diagnostics.csv",
        ))
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise PredictionError(
            "Required locally produced artifacts are missing. "
            "Do not retrain or rerun the frozen test just to open this app. Missing: "
            + ", ".join(missing[:6])
        )
    return tuple((name, (root / name).stat().st_size, (root / name).stat().st_mtime_ns) for name in files)


def _render_result(out: dict, bundle) -> None:
    st.subheader("Molecular prediction")
    top_left, top_right = st.columns([1, 1.3], gap="large")
    with top_left:
        st.image(Draw.MolToImage(Chem.MolFromSmiles(out["canonical_smiles"]), size=(450, 320)))
        st.caption(f"Canonical SMILES: `{out['canonical_smiles']}`")
    with top_right:
        first, second = st.columns(2)
        first.metric("Predicted logD at pH 7.4", f"{out['predicted_logd']:.2f}")
        second.metric("Closest training similarity", f"{out['max_train_tanimoto']:.3f}")
        st.write(f"**Similarity:** {out['similarity_band']}")
        st.caption(out["similarity_explanation"])
        st.progress(min(1.0, max(0.0, out["max_train_tanimoto"])))
        st.write(f"**Closest training compound:** `{out['nearest_train_compound_id']}`")

    st.divider()
    st.subheader("Separately calibrated prediction interval")
    st.write(
        f"**Nominal {100 * out['nominal_interval_coverage']:.0f}% interval:** "
        f"[{out['prediction_interval_lower']:.2f}, {out['prediction_interval_upper']:.2f}] logD"
    )
    st.caption(
        f"Interval centre = {out['calibration_model_prediction_logd']:.2f} logD "
        f"from a **separate proper-training model**, not the primary {out['predicted_logd']:.2f} estimate. "
        f"On the scaffold-held-out test, this interval procedure covered "
        f"{out['scaffold_test_interval_coverage'] * 100:.1f}% of molecules. "
        "Nominal coverage is not guaranteed for unfamiliar chemistry."
    )

    for warning in out["warnings"]:
        st.warning(warning)

    with st.expander("Molecular descriptors and nearest-neighbour details"):
        st.dataframe(
            [{"Descriptor": k, "Computed value": v} for k, v in out["molecular_descriptors"].items()],
            hide_index=True, width="stretch",
        )
        st.caption("RDKit MolLogP is a calculated logP descriptor, not the model's experimental logD target.")
        st.write("**Nearest training SMILES**")
        st.code(out["nearest_train_canonical_smiles"])

    export = {k: v for k, v in out.items() if k != "molecular_descriptors"}
    st.download_button(
        "Download prediction details (JSON)",
        data=json.dumps(export, indent=2, sort_keys=True) + "\n",
        file_name="logd_prediction.json",
        mime="application/json",
    )



def main() -> None:
    st.title("Molecular Lipophilicity Explorer")
    st.caption("Experimental logD at pH 7.4 · Frozen scientific ML model · Local demonstration")
    st.info(
        "This software predicts molecular lipophilicity from structure. It does **not** predict "
        "drug efficacy, safety, toxicity, clinical outcomes, or suitability for synthesis."
    )
    with st.sidebar:
        st.subheader("Model provenance")
        st.write("**Data:** MoleculeNet Lipophilicity (AstraZeneca/ChEMBL)")
        st.write("**Primary model:** Morgan fingerprint Ridge (`alpha=10`)")
        st.write("**Primary evaluation:** Scaffold-held-out")
        st.caption("Only models you trained and saved locally should be loaded; joblib is pickle-based.")

    try:
        current_artifacts = _artifact_signature(PROJECT)
        bundle = _load_models(str(PROJECT), current_artifacts)
    except (PredictionError, OSError, ValueError) as exc:
        st.error(f"Could not verify your frozen local model files: {exc}")
        st.caption("Keep your existing models and reports in their original folders. See docs/prediction_app.md.")
        st.stop()

    with st.sidebar:
        st.divider()
        st.metric("Scaffold-test RMSE", f"{bundle.test_rmse:.3f} logD")
        st.metric("Scaffold-test MAE", f"{bundle.test_mae:.3f} logD")
        st.metric("Scaffold-test R²", f"{bundle.test_r2:.3f}")
        st.caption(f"Frozen scikit-learn {bundle.sklearn_version}")

    single, batch = st.tabs(["Single molecule", "CSV batch"])
    with single:
        with st.form("smiles_form"):
            text = st.text_input("Molecular SMILES", placeholder="Example: CCO", max_chars=1000)
            submitted = st.form_submit_button("Predict logD", type="primary")
        if submitted:
            try:
                st.session_state["molprop_prediction"] = (current_artifacts, predict_smiles(bundle, text))
            except PredictionError as exc:
                st.session_state.pop("molprop_prediction", None)
                st.error(str(exc))
        if ("molprop_prediction" in st.session_state
                and st.session_state["molprop_prediction"][0] == current_artifacts):
            _render_result(st.session_state["molprop_prediction"][1], bundle)
        else:
            st.caption("Enter one SMILES string and select Predict logD. The input stays on your machine.")

    with batch:
        st.write("Upload a local CSV with a `smiles` column and optionally `compound_id`. Maximum 100 rows.")
        file = st.file_uploader("SMILES CSV", type="csv", accept_multiple_files=False)
        file_hash = hashlib.sha256(file.getvalue()).hexdigest() if file is not None else None
        if st.button("Run batch inference", disabled=file is None):
            try:
                source_rows = decode_smiles_csv(file.getvalue()) if file is not None else []
                st.session_state["molprop_batch"] = ((current_artifacts, file_hash), predict_batch_rows(bundle, source_rows))
            except PredictionError as exc:
                st.session_state.pop("molprop_batch", None)
                st.error(str(exc))
        if "molprop_batch" in st.session_state and st.session_state["molprop_batch"][0] == (current_artifacts, file_hash):
            result = st.session_state["molprop_batch"][1]
            ok = sum(row["status"] == "OK" for row in result)
            st.caption(f"{ok} valid molecular predictions out of {len(result)} input rows.")
            st.dataframe(result, width="stretch", hide_index=True)
            st.download_button(
                "Download predictions (CSV)", data=prediction_csv_bytes(result),
                file_name="molecular_predictions.csv", mime="text/csv",
            )

    st.divider()
    st.caption(
        "Similarity bands are descriptive, not a validated applicability domain. "
        "The held-out test results were calculated once using frozen models; "
        "this app does not retrain or tune them. Predictions are exploratory only."
    )


if __name__ == "__main__":
    main()
