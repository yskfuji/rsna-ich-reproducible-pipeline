#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_seeds(s: str) -> list[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty seeds")
    return vals


def main() -> int:
    p = argparse.ArgumentParser(description="Run audit_rsna_split.py for multiple seeds.")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)
    p.add_argument("--rsna-root", required=True, type=str)
    p.add_argument("--preprocessed-root", required=True, type=str)
    p.add_argument("--limit-images", default=8000, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)
    p.add_argument("--split-by", default="study", type=str)
    p.add_argument("--out-json", default="/tmp/rsna_split_audit_seed0to9.json", type=str)
    ns = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    audit = repo / "tools" / "audit_rsna_split.py"

    rows = []
    for seed in parse_seeds(ns.seeds):
        cmd = [
            sys.executable,
            str(audit),
            "--rsna-root",
            str(ns.rsna_root),
            "--preprocessed-root",
            str(ns.preprocessed_root),
            "--limit-images",
            str(int(ns.limit_images)),
            "--val-frac",
            str(float(ns.val_frac)),
            "--split-by",
            str(ns.split_by),
            "--seed",
            str(seed),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode not in (0, 2):
            raise RuntimeError(f"seed={seed} failed rc={cp.returncode}\nstdout={cp.stdout}\nstderr={cp.stderr}")
        rep = json.loads(cp.stdout)
        rows.append(
            {
                "seed": seed,
                "n_group_intersection": int(rep["n_group_intersection"]),
                "n_imageid_intersection": int(rep["n_imageid_intersection"]),
                "n_train": int(rep["n_train"]),
                "n_val": int(rep["n_val"]),
            }
        )

    summary = {
        "all_zero_group_intersection": all(r["n_group_intersection"] == 0 for r in rows),
        "all_zero_imageid_intersection": all(r["n_imageid_intersection"] == 0 for r in rows),
        "max_group_intersection": max(r["n_group_intersection"] for r in rows),
        "max_imageid_intersection": max(r["n_imageid_intersection"] for r in rows),
        "rows": rows,
    }

    out = Path(str(ns.out_json)).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
