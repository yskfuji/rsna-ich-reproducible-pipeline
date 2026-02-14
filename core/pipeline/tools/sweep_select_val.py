"""Run val-only conditional-cascade evaluation for multiple experiments and select the best.

This script is designed to be robust to terminal disconnects by writing per-run logs.

Selection rule (default): maximize average of (gt_3mm mean_dice, le_3mm mean_dice).

Usage example:
  /opt/anaconda3/envs/medseg_unet/bin/python tools/sweep_select_val.py \
    --experiments \
      medseg_3d_unet_stage2_conditional_residual_sweep_w010_nl025_cu025 \
      medseg_3d_unet_stage2_conditional_residual_sweep_w010_nl050_cu025 \
      medseg_3d_unet_stage2_conditional_residual_sweep_w025_nl025_cu025 \
      medseg_3d_unet_stage2_conditional_residual_sweep_w025_nl050_cu025
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _best_row(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("per_threshold", [])
    if not rows:
        raise ValueError("summary.json has no per_threshold")
    row = max(rows, key=lambda d: float(d.get("mean_dice") or 0.0))
    by = row.get("by_slice_spacing", {}) or {}

    def _grp(name: str) -> dict[str, Any]:
        g = by.get(name, {}) or {}
        return {
            "mean_dice": float(g.get("mean_dice") or 0.0),
            "detection_rate_case": float(g.get("detection_rate_case") or 0.0),
            "n_cases": int(g.get("n_cases") or 0),
        }

    out = {
        "best_thr": float(row.get("threshold") or 0.0),
        "mean_dice": float(row.get("mean_dice") or 0.0),
        "gt_3mm": _grp("gt_3mm"),
        "le_3mm": _grp("le_3mm"),
    }
    out["score_avg_gtle"] = 0.5 * (out["gt_3mm"]["mean_dice"] + out["le_3mm"]["mean_dice"])
    return out


def _run_step(repo: Path, out_dir: Path, model: Path, step: str, env_base: dict[str, str], log_path: Path) -> None:
    script = repo / "tools" / "run_conditional_cascade_eval.sh"
    cmd = ["bash", str(script), step]
    env = dict(env_base)
    env.update(
        {
            "OUT": str(out_dir),
            "MODEL": str(model),
        }
    )

    with log_path.open("ab") as f:
        f.write(("\n[cmd] " + " ".join(cmd) + "\n").encode("utf-8"))
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"step={step} failed rc={rc} (see {log_path})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repo root (ToReBrain-pipeline)",
    )
    ap.add_argument("--python", default=sys.executable, help="Python executable (used for reporting only)")
    ap.add_argument("--experiments", nargs="+", required=True, help="runs/3d_unet/<experiment> names")
    ap.add_argument(
        "--out-root",
        default=None,
        help="Base output dir under results/diag (default: results/diag/sweep_val_select_<timestamp>)",
    )
    ap.add_argument("--fusion", default="residual", choices=["max", "residual"])
    ap.add_argument("--stage1-logit-eps", default="1e-4")
    ap.add_argument("--resample-max-zoom-mm", default="2.0")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_root = (
        repo / "results" / "diag" / (args.out_root or f"sweep_val_select_{_timestamp()}")
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env_base = dict(os.environ)
    env_base["PYTHONPATH"] = str(repo)
    env_base.setdefault("PATCH_SIZE", "56,56,24")
    env_base.setdefault("OVERLAP", "0.5")
    env_base.setdefault("NORM", "nonzero_zscore")
    env_base.setdefault("TTA", "none")
    env_base["FUSION"] = args.fusion
    env_base["STAGE1_LOGIT_EPS"] = str(args.stage1_logit_eps)
    env_base["RESAMPLE_MAX_ZOOM_MM"] = str(args.resample_max_zoom_mm)

    results: list[dict[str, Any]] = []

    for exp in args.experiments:
        model = repo / "runs" / "3d_unet" / exp / "best.pt"
        if not model.exists():
            raise FileNotFoundError(str(model))

        out_dir = out_root / exp
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"

        # If eval already exists, reuse.
        summary_path = out_dir / "eval_val" / "summary.json"
        if not summary_path.exists():
            _run_step(repo, out_dir, model, "val", env_base, log_path)
            _run_step(repo, out_dir, model, "eval_val", env_base, log_path)

        best = _best_row(_read_json(summary_path))
        best["experiment"] = exp
        best["model"] = str(model)
        best["out_dir"] = str(out_dir)
        results.append(best)

    results_sorted = sorted(results, key=lambda r: (r["score_avg_gtle"], r["mean_dice"]), reverse=True)

    out_json = out_root / "summary_val_ranked.json"
    out_json.write_text(json.dumps({"ranked": results_sorted}, indent=2, ensure_ascii=False) + "\n")

    best = results_sorted[0]
    print(f"[out] {out_root}")
    print(f"[best] exp={best['experiment']} score_avg_gtle={best['score_avg_gtle']:.5f} mean={best['mean_dice']:.5f} thr={best['best_thr']:.2f}")
    print(f"       gt_3mm={best['gt_3mm']['mean_dice']:.5f} le_3mm={best['le_3mm']['mean_dice']:.5f}")
    print(f"       model={best['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
