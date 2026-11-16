"""Download contract tested without making a live network request."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from molprop.download import DownloadError, check_csv_bytes, fetch_csv


def test_csv_payload_contract():
    payload = b"CMPD_CHEMBLID,exp,smiles\nTEST1,2.1,CCO\n"
    assert check_csv_bytes(payload) == 1
    with pytest.raises(DownloadError, match="expected lipophilicity CSV"):
        check_csv_bytes(b"not,csv\nA,B\n")
    with pytest.raises(DownloadError, match="empty"):
        check_csv_bytes(b"smiles,exp\n")
    with pytest.raises(DownloadError, match="UTF-8"):
        check_csv_bytes(b"\xff\xfe")


def test_fetch_records_hash_and_does_not_overwrite(tmp_path: Path, monkeypatch):
    raw = b"CMPD_CHEMBLID,exp,smiles\nTEST1,2.1,CCO\n"
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n):
            data = self.payload
            self.payload = b""
            return data
        def __init__(self): self.payload = raw
    monkeypatch.setattr("molprop.download.urlopen", lambda *args, **kwargs: FakeResponse())
    destination = tmp_path / "raw" / "Lipophilicity.csv"
    m = fetch_csv(destination)
    assert m["rows"] == 1
    assert m["sha256"] == hashlib.sha256(raw).hexdigest()
    assert destination.read_bytes() == raw
    assert destination.with_name("source_manifest.json").exists()
    with pytest.raises(DownloadError, match="exists"):
        fetch_csv(destination)
