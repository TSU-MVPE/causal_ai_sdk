"""Unit test: SDK test_data must match source of truth (MVP/scripts/test_data).

Source of truth is MVP/scripts/test_data. The SDK test_data directory must have
the same set of files with identical content (SHA256) so the two stay in sync.
No allowlist; any difference fails the test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_files_and_hashes(root: Path) -> dict[str, str]:
    out = {}
    for f in root.rglob("*"):
        if f.is_file():
            rel = f.relative_to(root).as_posix()
            out[rel] = _file_sha256(f)
    return out


def test_sdk_test_data_matches_source_of_truth():
    """SDK test_data must match MVP/scripts/test_data (file list + SHA256).

    Source of truth: MVP/scripts/test_data. Run from repo root (MVP) so paths resolve.

    Raises:
        AssertionError: If files or content differ between source of truth and SDK test_data.
    """
    this_file = Path(__file__).resolve()
    mvp_root = this_file.parents[4]
    source_of_truth = mvp_root / "scripts" / "test_data"
    sdk_test_data = mvp_root / "causal_ai_sdk" / "test_data"

    source_hashes = _rel_files_and_hashes(source_of_truth)
    sdk_hashes = _rel_files_and_hashes(sdk_test_data)

    missing_in_sdk = sorted(set(source_hashes.keys()) - set(sdk_hashes.keys()))
    if missing_in_sdk:
        raise AssertionError(
            f"Files in source of truth (MVP/scripts/test_data) missing in SDK test_data: "
            f"{missing_in_sdk}. Keep SDK test_data in sync with scripts/test_data."
        )

    extra_in_sdk = sorted(set(sdk_hashes.keys()) - set(source_hashes.keys()))
    if extra_in_sdk:
        raise AssertionError(
            f"Files in SDK test_data not in source of truth: {extra_in_sdk}. "
            "SDK test_data must match scripts/test_data exactly (no extra files)."
        )

    mismatched = []
    for rel in source_hashes:
        if source_hashes[rel] != sdk_hashes[rel]:
            mismatched.append(rel)
    if mismatched:
        raise AssertionError(
            f"Content mismatch (SHA256) between source of truth and SDK test_data: "
            f"{sorted(mismatched)}. Keep SDK test_data in sync with scripts/test_data."
        )
