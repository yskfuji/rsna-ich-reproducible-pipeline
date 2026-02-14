#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty seeds list")
    return out


def _nan_stats(xs: list[float]) -> dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


@dataclass(frozen=True)
class RunSpec:
    seed: int
    mc_seed: int
    out_curve_png: Path
    out_reliability_png: Path


def _run_one(
    *,
    eval_script: Path,
    base_args: list[str],
    run: RunSpec,
    env: dict[str, str] | None,
) -> dict[str, Any]:
    args = [sys.executable, str(eval_script)] + base_args + [
        "--seed",
        str(run.seed),
        "--mc-seed",
        str(run.mc_seed),
        "--out-curve-png",
        str(run.out_curve_png),
        "--out-reliability-png",
        str(run.out_reliability_png),
    ]
    cp = subprocess.run(args, capture_output=True, text=True, env=env)
    if cp.returncode != 0:
        raise RuntimeError(
            "eval script failed\n"
            f"cmd: {' '.join(args)}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}\n"
        )
    try:
        return json.loads(cp.stdout)
    except Exception as e:
        raise RuntimeError(f"failed to parse JSON from stdout: {e}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run tools/eval_rsna_uncertainty.py for multiple seeds and summarize mean±std/min/max. "
            "This evaluates holdout split variability (and MC-dropout randomness if mc_seed varies)."
        )
    )
    p.add_argument("--seeds", default="0,1,2", type=str, help="Comma-separated split seeds (default 0,1,2)")
    p.add_argument(
        "--mc-seeds",
        default="same",
        type=str,
        help="'same' to set mc_seed=seed, or comma-separated list matching --seeds length, or a single int",
    )
    p.add_argument(
        "--out-dir",
        default="results/uncertainty/sweeps",
        type=str,
        help="Directory to store per-seed PNGs",
    )

    # Pass-through: everything after '--' goes to eval script.
    p.add_argument(
        "--",
        dest="sep",
        action="store_true",
        help="Separator; arguments after this are passed to eval_rsna_uncertainty.py",
    )
    p.add_argument("eval_args", nargs=argparse.REMAINDER)

    ns = p.parse_args(argv)

    seeds = _parse_int_list(ns.seeds)

    if str(ns.mc_seeds).strip().lower() == "same":
        mc_seeds = list(seeds)
    else:
        mc_s = str(ns.mc_seeds).strip()
        if "," in mc_s:
            mc_seeds = _parse_int_list(mc_s)
            if len(mc_seeds) != len(seeds):
                raise SystemExit("--mc-seeds length must match --seeds when comma-separated")
        else:
            v = int(mc_s)
            mc_seeds = [v for _ in seeds]

    out_dir = Path(str(ns.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate eval script
    eval_script = Path(__file__).resolve().parent / "eval_rsna_uncertainty.py"
    if not eval_script.exists():
        raise SystemExit(f"missing eval script: {eval_script}")

    # Remove leading '--' if present.
    base_args = list(ns.eval_args)
    if base_args and base_args[0] == "--":
        base_args = base_args[1:]

    # Propagate TORCH_DEVICE if set.
    env = None
    if "TORCH_DEVICE" in os.environ:
        env = dict(os.environ)

    per_seed: list[dict[str, Any]] = []

    for s, ms in zip(seeds, mc_seeds, strict=False):
        run = RunSpec(
            seed=int(s),
            mc_seed=int(ms),
            out_curve_png=out_dir / f"coverage_risk_seed{s}.png",
            out_reliability_png=out_dir / f"reliability_any_seed{s}.png",
        )
        out = _run_one(eval_script=eval_script, base_args=base_args, run=run, env=env)
        out["seed"] = int(s)
        out["mc_seed"] = int(ms)
        per_seed.append(out)

    keys = [
        "temperature",
        "ece_any",
        "brier_any",
        "brier_weighted",
        "nll_weighted_logloss",
        "auroc_uncertainty_detect_error_any",
        "aurc_weighted_logloss",
        "accuracy_any_full",
        "accuracy_any_at_coverage",
        "accuracy_any_improve_pp",
    ]

    summary: dict[str, Any] = {
        "seeds": list(seeds),
        "mc_seeds": list(mc_seeds),
        "n_runs": int(len(per_seed)),
        "per_seed": per_seed,
        "summary": {},
    }

    for k in keys:
        xs: list[float] = []
        for r in per_seed:
            v = r.get(k)
            if isinstance(v, (int, float)):
                xs.append(float(v))
        summary["summary"][k] = _nan_stats(xs)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
