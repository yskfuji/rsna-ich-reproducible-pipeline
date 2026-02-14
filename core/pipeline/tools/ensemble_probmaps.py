"""Average multiple per-case probability maps (NPZ) into a single probs directory.

Expected input format:
- Each input directory contains <case_id>.npz with key 'probs' (Z,Y,X) float16/float32.

Typical workflow:
1) For each trained model, run evaluate_isles with --save-probs to generate out_dir/probs
2) Average them with this script
3) Evaluate the ensemble using evaluate_isles with --probs-dir (and any thresholds/postprocess)

Example:
  python tools/ensemble_probmaps.py \
    --probs-dirs runsA/probs runsB/probs runsC/probs \
    --out-probs-dir results/diag/ensemble_20251225/probs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dirs", nargs="+", required=True, help="Input dirs containing <case_id>.npz")
    ap.add_argument("--out-probs-dir", required=True, help="Output directory for averaged <case_id>.npz")
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Output dtype",
    )
    args = ap.parse_args()

    in_dirs = [Path(p).expanduser().resolve() for p in args.probs_dirs]
    for d in in_dirs:
        if not d.exists():
            raise FileNotFoundError(str(d))

    out_dir = Path(args.out_probs_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    first = in_dirs[0]
    case_paths = sorted(first.glob("*.npz"))
    if not case_paths:
        raise RuntimeError(f"No npz files found in: {first}")

    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    n = 0
    for p0 in case_paths:
        case_id = p0.stem
        probs_list = []
        shape = None
        for d in in_dirs:
            p = d / f"{case_id}.npz"
            if not p.exists():
                raise FileNotFoundError(f"Missing {case_id}.npz in {d}")
            with np.load(str(p)) as z:
                a = z["probs"].astype(np.float32, copy=False)
            if shape is None:
                shape = a.shape
            elif a.shape != shape:
                raise ValueError(f"Shape mismatch for case_id={case_id}: {a.shape} vs {shape}")
            probs_list.append(a)

        avg = np.mean(np.stack(probs_list, axis=0), axis=0).astype(out_dtype, copy=False)
        np.savez_compressed(str(out_dir / f"{case_id}.npz"), probs=avg)
        n += 1

    print(f"[done] wrote {n} cases to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
