#!/usr/bin/env python3
"""Import every module under core/pipeline/{src,tools} from a fresh clone.

A local working tree can contain files that Git ignores (this is how
src/datasets/ went unpublished), so by default the check runs against a
temporary `git clone` of HEAD. Use --in-place when the checkout is already
clean, e.g. in CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("src", "tools")

_PROBE = r"""
import importlib, pathlib, sys
sys.path.insert(0, ".")
mods = set()
for pkg in sys.argv[1:]:
    for p in pathlib.Path(pkg).rglob("*.py"):
        name = ".".join(p.with_suffix("").parts)
        mods.add(name.removesuffix(".__init__"))
failed = []
for m in sorted(mods):
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append(f"{m}: {e!r}")
print(f"imported {len(mods) - len(failed)}/{len(mods)} modules")
for f in failed:
    print(f"FAILED {f}")
sys.exit(1 if failed else 0)
"""


def _import_all(pipeline_dir: Path) -> int:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, *PACKAGES],
        cwd=pipeline_dir,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-place", action="store_true", help="Check the current checkout instead of a fresh clone.")
    ns = ap.parse_args()

    if ns.in_place:
        return _import_all(ROOT / "core/pipeline")

    with tempfile.TemporaryDirectory(prefix="rsna-clean-") as tmp:
        clone = Path(tmp) / "repo"
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(ROOT), str(clone)], check=True)
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, text=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "-q", head.stdout.strip()], check=True)
        return _import_all(clone / "core/pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
