"""Fetch the published MoleculeNet lipophilicity CSV without committing raw data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
DOI = "10.6019/chembl3301361"
MAX_BYTES = 10_000_000  # Fail closed if an unexpected response is served.


class DownloadError(RuntimeError):
    """Source retrieval or file validation failed."""


def check_csv_bytes(payload: bytes) -> int:
    try:
        contents = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(contents))
        if reader.fieldnames is None or not {"smiles", "exp"}.issubset(reader.fieldnames):
            raise DownloadError("Downloaded content is not the expected lipophilicity CSV")
        count = sum(1 for _ in reader)
    except UnicodeDecodeError as exc:
        raise DownloadError("Downloaded data is not UTF-8 CSV") from exc
    if count < 1:
        raise DownloadError("Downloaded CSV is empty")
    return count


def fetch_csv(destination: Path, *, force: bool = False) -> dict:
    """Atomically download and hash the source; never silently overwrite it."""
    destination = Path(destination)
    manifest_file = destination.with_name("source_manifest.json")
    if destination.exists() and not force:
        raise DownloadError(f"Source already exists: {destination}. Refusing to overwrite. Use --force deliberately.")
    request = Request(DATA_URL, headers={"User-Agent": "MolecularPropertyPrediction/0.1 (research reproducibility)"})
    try:
        with urlopen(request, timeout=45) as response:
            size = 0
            chunks = []
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise DownloadError("Download exceeded the expected 10 MB limit")
                chunks.append(chunk)
        payload = b"".join(chunks)
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"Download failed: {exc}") from exc
    rows = check_csv_bytes(payload)
    manifest = {
        "dataset": "MoleculeNet Lipophilicity (AstraZeneca/ChEMBL)",
        "source_url": DATA_URL,
        "source_doi": DOI,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_file": destination.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": rows,
        "expected_original_rows": 4200,
        "raw_data_redistributed": False,
        "license_note": "Dataset redistribution terms not yet established; keep raw data local.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write raw data only after verifying the downloaded content.
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".lipo-", suffix=".tmp", delete=False) as temp:
        raw_tmp = Path(temp.name)
        temp.write(payload)
    try:
        os.replace(raw_tmp, destination)
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finally:
        raw_tmp.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and hash the public lipophilicity dataset")
    parser.add_argument("--output", type=Path, default=Path("data/raw/Lipophilicity.csv"))
    parser.add_argument("--force", action="store_true", help="Replace an existing local file")
    args = parser.parse_args()
    try:
        report = fetch_csv(args.output, force=args.force)
    except DownloadError as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
