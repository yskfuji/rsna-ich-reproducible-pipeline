#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

REPO_NAME = "rsna-ich-reproducible-pipeline"
REQUIRED_PATHS = [
    "README.md",
    "sample_manifest.json",
    "rsna_ich/README.md",
    "rsna_ich/README_en.md",
    "core/pipeline/README.md",
    "core/pipeline/README_en.md",
    "core/pipeline/tools/make_manifest.py",
    "core/pipeline/train_rsna_cnn2d_classifier.py",
    "core/pipeline/predict_rsna_ich_submission.py",
    "core/pipeline/serve_rsna_ich_api.py",
    "core/pipeline/src/inference/serve_rsna_ich_api.py",
    "core/pipeline/src/datasets/rsna_ich_dataset.py",
    "core/pipeline/requirements.txt",
    "core/pipeline/reports/rsna_cv_summary.json",
]
SOURCE_DIRS = ["core/pipeline/src", "core/pipeline/tools", "scripts"]
ENTRYPOINTS = [
    "core/pipeline/train_rsna_cnn2d_classifier.py train",
    "core/pipeline/predict_rsna_ich_submission.py",
    "core/pipeline/serve_rsna_ich_api.py",
    "core/pipeline/tools/make_manifest.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _check_paths(root: Path) -> list[str]:
    missing: list[str] = []
    for rel in REQUIRED_PATHS:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def _unpublished_sources(root: Path) -> list[str] | None:
    """Return .py files present on disk but not tracked by Git (None if not a Git checkout)."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", *SOURCE_DIRS],
            cwd=root,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    tracked = set(filter(None, proc.stdout.split("\0")))
    on_disk = {
        p.relative_to(root).as_posix()
        for d in SOURCE_DIRS
        for p in (root / d).rglob("*.py")
        if "__pycache__" not in p.parts
    }
    return sorted(on_disk - tracked)


def _load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "sample_manifest.json").read_text(encoding="utf-8"))


def _run_manifest(root: Path, out_dir: Path) -> Path:
    out_path = out_dir / "MANIFEST.sha256.txt"
    cmd = [
        sys.executable,
        str(root / "core/pipeline/tools/make_manifest.py"),
        "--root",
        str(root),
        "--out",
        str(out_path.relative_to(root)),
        "--exclude",
        "artifacts/smoke_dummy/**",
    ]
    subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a no-data smoke test for the public RSNA repository.")
    parser.add_argument("--use_dummy_data", action="store_true", help="Load the bundled dummy manifest and generate a smoke-test summary.")
    args = parser.parse_args()

    root = _repo_root()
    missing = _check_paths(root)
    if missing:
        print("Missing required files:", *missing, sep="\n- ")
        return 1

    unpublished = _unpublished_sources(root)
    if unpublished:
        print("Source files exist locally but are not tracked by Git (check .gitignore):", *unpublished, sep="\n- ")
        return 1

    out_dir = root / "artifacts" / "smoke_dummy"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = _load_manifest(root) if args.use_dummy_data else {"sample_cases": []}
    manifest_out = _run_manifest(root, out_dir)

    summary: dict[str, Any] = {
        "repo": REPO_NAME,
        "status": "passed",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "used_dummy_data": bool(args.use_dummy_data),
        "checked_paths": REQUIRED_PATHS,
        "git_tracking_checked": unpublished is not None,
        "entrypoints": ENTRYPOINTS,
        "sample_cases": manifest.get("sample_cases", []),
        "generated_files": [
            str(manifest_out.relative_to(root)),
            "artifacts/smoke_dummy/summary.json",
        ],
        "notes": [
            "This smoke test validates repository wiring without bundling protected medical data.",
            "Use the detailed README files for full training and evaluation commands.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[{REPO_NAME}] smoke test passed")
    print(f"- dummy data mode: {args.use_dummy_data}")
    print(f"- manifest: {manifest_out.relative_to(root)}")
    print("- next: open README.md for quickstart or core/pipeline/README_en.md for full reproduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
