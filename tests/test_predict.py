"""Tests of local inference without using real held-out test labels."""
from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn import __version__ as sklearn_version

from molprop.final_evaluation import final_evaluation
from molprop.predict import (
    FrozenBundle, PredictionError, _descriptive_similarity_band,
    _validate_final_report, canonicalize_input, load_frozen_bundle, predict_smiles,
)
from molprop.prepare import digest
from molprop.train import load_prepared

from test_analyze import ready  # noqa: F401 -- fictional synthetic fixture
from test_final_evaluation import locked_toy  # noqa: F401 -- fictional fixture

ROOT = Path(__file__).resolve().parents[1]


class ConstantPredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.repeat(self.value, x.shape[0])


def test_invalid_smiles_and_limits():
    for bad in ("", " ", "C1CCC", "not-smiles", "C" * 1001, "C\nC", "C\tC"):
        with pytest.raises(PredictionError):
            canonicalize_input(bad)
    with pytest.raises(PredictionError):
        canonicalize_input(None)


def test_canonical_smiles_preserves_stereochemistry():
    canonical, mol = canonicalize_input(" C[C@H](N)C(=O)O ")
    assert "@" in canonical
    assert mol.GetNumAtoms() > 0


def fake_bundle() -> FrozenBundle:
    generation = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    smiles = ("CCO", "c1ccccc1", "CCN")
    fps = tuple(generation.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles)
    return FrozenBundle(
        full_model=ConstantPredictor(2.2), calibration_model=ConstantPredictor(1.6),
        calibration_quantile=1.5, nominal_coverage=0.9, test_coverage=0.885714,
        test_rmse=0.923454, test_mae=0.73277, test_r2=0.411915,
        train_fingerprints=fps, train_identifiers=("A", "B", "C"), train_smiles=smiles,
        radius=2, bits=2048, include_chirality=True,
        strategy="scaffold_held_out", sklearn_version=sklearn_version,
    )


def test_prediction_retains_distinct_interval_model_and_training_neighbor():
    result = predict_smiles(fake_bundle(), "CCO")
    assert result["predicted_logd"] == 2.2
    assert result["calibration_model_prediction_logd"] == 1.6
    assert result["prediction_interval_lower"] == 0.1
    assert result["prediction_interval_upper"] == 3.1
    assert result["max_train_tanimoto"] == 1.0
    assert result["nearest_train_compound_id"] == "A"
    assert len(result["molecular_descriptors"]) == 15
    assert any("DIFFERENT calibration model" in x for x in result["warnings"])


def test_mixture_and_charged_structure_warnings():
    result = predict_smiles(fake_bundle(), "[NH4+].[Cl-]")
    assert any("disconnected" in x for x in result["warnings"])
    # Two components carry opposite charges, so global charge is zero.
    charged = predict_smiles(fake_bundle(), "[NH4+]")
    assert any("Charged" in x for x in charged["warnings"])


def test_similarity_band_boundaries():
    assert _descriptive_similarity_band(0.0)[0] == "Very low similarity"
    assert _descriptive_similarity_band(0.3)[0] == "Low similarity"
    assert _descriptive_similarity_band(0.5)[0] == "Moderate similarity"
    assert _descriptive_similarity_band(0.7)[0] == "Higher similarity"


@pytest.fixture(scope="module")
def complete_fictional_run(locked_toy, tmp_path_factory):
    processed, baseline, analysis, analysis_config, lock = locked_toy
    final = tmp_path_factory.mktemp("frozen-app") / "final"
    report = final_evaluation(
        processed, baseline, analysis, final, lock, ROOT / "configs/models.json", analysis_config
    )
    def kwargs(**updates):
        return dict(processed_dir=processed, baseline_dir=baseline, analysis_dir=analysis,
                    final_report_path=final / "final_report.json", lock_path=lock,
                    model_config=ROOT / "configs/models.json", analysis_config=analysis_config,
                    preparation_config=ROOT / "configs/preparation.json") | updates
    return report, kwargs


def test_frozen_bundle_accepts_locally_fitted_fictional_models(complete_fictional_run):
    report, kwargs = complete_fictional_run
    bundle = load_frozen_bundle(**kwargs())
    assert bundle.strategy == "scaffold_held_out"
    assert len(bundle.train_identifiers) == len(bundle.train_fingerprints)
    _, arrays, assignments = load_prepared(kwargs()["processed_dir"])
    assert len(bundle.train_identifiers) == sum(r["split"] == "train" for r in assignments[bundle.strategy])
    example = next(r["canonical_smiles"] for r in assignments[bundle.strategy] if r["split"] == "train")
    result = predict_smiles(bundle, example)
    assert result["max_train_tanimoto"] == 1.0
    assert result["prediction_interval_upper"] > result["prediction_interval_lower"]


def test_replaced_model_fails_integrity_before_joblib_loading(complete_fictional_run, tmp_path):
    report, kwargs = complete_fictional_run
    old = kwargs()["baseline_dir"]
    changed = tmp_path / "altered_baselines"
    shutil.copytree(old, changed)
    path = changed / "scaffold_held_out" / "ridge_morgan.joblib"
    with path.open("ab") as out:
        out.write(b"unexpected bytes")
    with pytest.raises(PredictionError, match="checksum mismatch"):
        load_frozen_bundle(**kwargs(baseline_dir=changed))


def test_frozen_final_report_tampering_rejected(complete_fictional_run, tmp_path):
    report, kwargs = complete_fictional_run
    modified = json.loads(kwargs()["final_report_path"].read_text())
    modified["primary_model"] = "random_forest_descriptors"
    file = tmp_path / "fake_final_report.json"
    file.write_text(json.dumps(modified))
    with pytest.raises(PredictionError, match="disagrees with frozen protocol"):
        load_frozen_bundle(**kwargs(final_report_path=file))


def test_saved_python_version_is_checked_before_deserialisation(complete_fictional_run, tmp_path):
    report, kwargs = complete_fictional_run
    lock = json.loads(kwargs()["lock_path"].read_text())
    lock["expected_sklearn_version"] = "0.0.0"
    other = tmp_path / "wrong_lock.json"
    other.write_text(json.dumps(lock))
    with pytest.raises(PredictionError, match="disagrees with frozen protocol"):
        load_frozen_bundle(**kwargs(lock_path=other))


def test_prediction_path_does_not_train_or_select_test_labels():
    from molprop import predict
    source = inspect.getsource(predict)
    assert ".fit(" not in source
    assert 'arrays["logd"]' not in source
    assert "GridSearchCV" not in source


def test_csv_batch_parsing_and_bounds():
    from molprop.predict import decode_smiles_csv
    records = decode_smiles_csv(b"compound_id,smiles\nsmall,CCO\nring,c1ccccc1\n")
    assert records == [("small", "CCO"), ("ring", "c1ccccc1")]
    assert decode_smiles_csv(b"smiles\nCCO\n") == [("row-1", "CCO")]
    for payload in (b"smiles\n", b"compound_id\n1\n", b"smiles,smiles\nCCO,CCO\n", b"smiles\n" + b"CCO\n" * 101):
        with pytest.raises(PredictionError):
            decode_smiles_csv(payload)
    with pytest.raises(PredictionError):
        decode_smiles_csv(b"smiles\n" + b"a" * 200_001)


def test_batch_prediction_reports_invalid_rows_individually():
    from molprop.predict import predict_batch_rows
    rows = predict_batch_rows(fake_bundle(), [("good", "CCO"), ("bad", "not-smiles")])
    assert len(rows) == 2
    assert rows[0]["status"] == "OK"
    assert rows[1]["status"].startswith("INVALID:")
    assert rows[0]["interval_lower"] != rows[0]["predicted_logd"]
    for wrong in ([], [("x", "CCO")] * 101):
        with pytest.raises(PredictionError):
            predict_batch_rows(fake_bundle(), wrong)


def test_batch_csv_escapes_formula_like_identifiers():
    import csv
    import io
    from molprop.predict import prediction_csv_bytes
    payload = prediction_csv_bytes([{"compound_id": "=DANGER()", "smiles": "CCO", "status": "OK"}])
    parsed = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    assert parsed[0]["compound_id"] == "'=DANGER()"
    assert parsed[0]["smiles"] == "CCO"
