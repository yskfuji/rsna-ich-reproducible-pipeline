from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _run(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)


def test_dataset_module_is_importable() -> None:
    from src.datasets.rsna_ich_dataset import RSNA_CLASSES, compute_subset_fingerprint_sha256

    assert RSNA_CLASSES[-1] == "any"
    assert len(RSNA_CLASSES) == 6
    assert compute_subset_fingerprint_sha256(["b", "a", "a", ""]) == compute_subset_fingerprint_sha256(["a", "b"])


def test_metric_claims_match_summary() -> None:
    proc = _run("scripts/check_metric_claims.py")
    assert proc.returncode == 0, proc.stdout


def test_public_hygiene() -> None:
    proc = _run("scripts/check_public_hygiene.py")
    assert proc.returncode == 0, proc.stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_smoke_test_rejects_untracked_source(tmp_path: Path) -> None:
    clone = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(REPO), str(clone)], check=True)
    (clone / "core/pipeline/src/untracked_probe.py").write_text("X = 1\n", encoding="utf-8")

    proc = _run("scripts/smoke_test.py", "--use_dummy_data", cwd=clone)

    assert proc.returncode == 1
    assert "core/pipeline/src/untracked_probe.py" in proc.stdout
