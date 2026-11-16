"""Browser-independent Streamlit smoke check; runs if the optional app is installed."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_explains_missing_artifacts_in_clean_checkout():
    # UI may load local frozen artifacts on the developer's machine; either
    # success or a clear missing-input state must be rendered without traceback.
    result = AppTest.from_file(str(ROOT / "src/molprop/app.py"), default_timeout=35).run()
    assert not result.exception
    assert len(result.title) >= 1
