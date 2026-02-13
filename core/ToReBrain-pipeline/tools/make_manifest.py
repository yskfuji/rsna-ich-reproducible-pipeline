#!/usr/bin/env python3
"""Generate a sha256 file manifest to fingerprint a snapshot/bundle.

Usage (from ToReBrain-pipeline/):
  python tools/make_manifest.py

This writes a deterministic, sorted manifest of (sha256, size, relative path).
By default it excludes large or non-essential directories (e.g., Datasets/) and
transient files (e.g., __pycache__/).

The manifest itself can be hashed to obtain a single digest for the bundle.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_GLOBS = [
    ".git/**",
    ".pytest_cache/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".DS_Store",
    "Datasets/**",
]


def _iter_files(root: Path, exclude_globs: list[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in exclude_globs):
            continue

        yield path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a sha256 manifest for this bundle")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to fingerprint (default: current directory)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/MANIFEST.sha256.txt"),
        help="Output manifest path (default: artifacts/MANIFEST.sha256.txt)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional exclude glob (can be repeated). Globs match POSIX relative paths.",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    out_path = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    exclude_globs = list(DEFAULT_EXCLUDE_GLOBS) + list(args.exclude)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# manifest_format: sha256 size_bytes path")
    lines.append(f"# root: {Path('.').resolve().name}")
    for file_path in _iter_files(root, exclude_globs):
        rel = file_path.relative_to(root).as_posix()
        size = file_path.stat().st_size
        digest = _sha256_file(file_path)
        lines.append(f"{digest} {size} {rel}")

    manifest_text = "\n".join(lines) + "\n"
    out_path.write_text(manifest_text, encoding="utf-8")

    print(f"Wrote: {out_path.relative_to(root) if out_path.is_relative_to(root) else out_path}")
    print(f"Files: {max(0, len(lines) - 2)}")
    print(f"Manifest sha256: {_sha256_text(manifest_text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
