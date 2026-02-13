"""Wait for trained models (best.pt) then run: save-probs -> ensemble -> sweep.

This script is designed for unstable terminals: run it via nohup.

Inputs are YAML training configs so we can infer:
- experiment_name -> runs/3d_unet/<experiment_name>/best.pt
- normalize mode for inference

Workflow:
1) For each config: wait until best.pt exists, then run evaluate_isles --save-probs
2) Average probs across models with tools/ensemble_probmaps.py
3) Run tools/postprocess_sweep.py on the ensemble via --probs-dir

Example:
  /opt/anaconda3/envs/medseg_unet/bin/python tools/run_saveprobs_ensemble_sweep.py \
    --python /opt/anaconda3/envs/medseg_unet/bin/python \
    --repo /Users/yusukefujinami/ToReBrain/ToReBrain-pipeline \
    --configs \
      configs/generated/_plan_20251224/medseg_3d_unet_e20_dwi_adc_flair_phase2_patch562424_pv2bg_ffg02_bgdist12_tversky_ohem_fgccinv_a1.yaml \
      configs/generated/_plan_20251225/medseg_3d_unet_e20_dwi_adc_flair_phase2_patch562424_pv2bg_ffg02_bgdist12_tversky_ohem_fgccinv_a1_norm_robustmad.yaml \
      configs/generated/_plan_20251225/medseg_3d_unet_e20_dwi_adc_flair_phase2_patch562424_pv2bg_ffg02_bgdist12_tversky_ohem_fgccinv_a1_norm_minmax01.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_yaml(p: Path) -> dict[str, Any]:
    return yaml.safe_load(p.read_text())


def _wait_for(path: Path, *, poll_s: float = 60.0) -> None:
    while not path.exists():
        print(f"[wait] {path} not found yet; sleeping {poll_s:.0f}s", flush=True)
        time.sleep(poll_s)


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True, help="Python executable")
    ap.add_argument("--repo", required=True, help="ToReBrain-pipeline root")
    ap.add_argument("--configs", nargs="+", required=True, help="Training YAML configs (including baseline zscore)")
    ap.add_argument("--csv-path", default="data/splits/my_dataset_dwi_adc_flair_train_val_test.csv")
    ap.add_argument("--root", default="data/processed/my_dataset_dwi_adc_flair")
    ap.add_argument("--split", default="test")
    ap.add_argument("--patch-size", default="56,56,24")
    ap.add_argument("--overlap", default="0.5")
    ap.add_argument(
        "--thresholds",
        default="0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.80",
    )
    ap.add_argument("--tta", default="flip")
    ap.add_argument("--resample-max-zoom-mm", default="2.0")
    ap.add_argument("--slice-spacing-source", default="raw")
    ap.add_argument(
        "--sweep-min-sizes",
        default="0,20,40",
        help="min_size grid for sweep",
    )
    ap.add_argument(
        "--sweep-top-ks",
        default="0,1,2",
        help="top_k grid for sweep",
    )
    ap.add_argument(
        "--sweep-cc-scores",
        default="none",
        help="cc_score grid for sweep (keep small unless needed)",
    )
    ap.add_argument(
        "--sweep-cc-score-thrs",
        default="0.5",
        help="cc_score_thr grid for sweep",
    )
    ap.add_argument("--focus-bucket", default="all")
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    py = str(Path(args.python).expanduser().resolve())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    stamp = _timestamp()
    diag_root = repo / "results" / "diag" / f"ensemble_pipeline_{stamp}"
    diag_root.mkdir(parents=True, exist_ok=True)

    probs_dirs: list[Path] = []

    # 1) Save probs per model
    for cfg_str in args.configs:
        cfg_path = Path(cfg_str)
        if not cfg_path.is_absolute():
            cfg_path = (repo / cfg_path).resolve()
        cfg = _load_yaml(cfg_path)
        exp = str(cfg.get("experiment_name"))
        norm = str((cfg.get("data", {}) or {}).get("normalize", "legacy_zscore"))

        best_pt = repo / "runs" / "3d_unet" / exp / "best.pt"
        _wait_for(best_pt, poll_s=float(args.poll_seconds))

        out_dir = diag_root / f"saveprobs_{exp}"
        probs_out = out_dir / "probs"

        # If already computed, skip.
        if probs_out.exists() and any(probs_out.glob("*.npz")):
            print(f"[skip] probs already exist: {probs_out}", flush=True)
        else:
            cmd = [
                py,
                "-m",
                "src.evaluation.evaluate_isles",
                "--model-path",
                str(best_pt),
                "--csv-path",
                str(args.csv_path),
                "--root",
                str(args.root),
                "--split",
                str(args.split),
                "--out-dir",
                str(out_dir),
                "--patch-size",
                str(args.patch_size),
                "--overlap",
                str(args.overlap),
                "--thresholds",
                str(args.thresholds),
                "--min-size",
                "0",
                "--top-k",
                "0",
                "--cc-score",
                "none",
                "--cc-score-thr",
                "0.5",
                "--normalize",
                norm,
                "--tta",
                str(args.tta),
                "--resample-max-zoom-mm",
                str(args.resample_max_zoom_mm),
                "--slice-spacing-source",
                str(args.slice_spacing_source),
                "--quiet",
                "--save-probs",
                "--save-probs-dtype",
                "float16",
            ]
            _run(cmd, cwd=repo, env=env)

        probs_dirs.append(probs_out)

    # 2) Ensemble average
    ens_dir = diag_root / "ensemble"
    ens_probs = ens_dir / "probs"
    if not (ens_probs.exists() and any(ens_probs.glob("*.npz"))):
        cmd = [
            py,
            "tools/ensemble_probmaps.py",
            "--probs-dirs",
            *[str(p) for p in probs_dirs],
            "--out-probs-dir",
            str(ens_probs),
            "--dtype",
            "float16",
        ]
        _run(cmd, cwd=repo, env=env)
    else:
        print(f"[skip] ensemble probs already exist: {ens_probs}", flush=True)

    # 3) Sweep on ensemble probs
    sweep_out = diag_root / "pp_sweep_ensemble"
    if not (sweep_out / "sweep.tsv").exists():
        cmd = [
            py,
            "tools/postprocess_sweep.py",
            "--probs-dir",
            str(ens_probs),
            "--csv-path",
            str(args.csv_path),
            "--root",
            str(args.root),
            "--split",
            str(args.split),
            "--out-root",
            str(sweep_out),
            "--patch-size",
            str(args.patch_size),
            "--overlap",
            str(args.overlap),
            "--normalize",
            "nonzero_zscore",
            "--tta",
            str(args.tta),
            "--thresholds",
            str(args.thresholds),
            "--min-sizes",
            str(args.sweep_min_sizes),
            "--top-ks",
            str(args.sweep_top_ks),
            "--cc-scores",
            str(args.sweep_cc_scores),
            "--cc-score-thrs",
            str(args.sweep_cc_score_thrs),
            "--resamples",
            str(args.resample_max_zoom_mm),
            "--slice-spacing-source",
            str(args.slice_spacing_source),
            "--focus-bucket",
            str(args.focus_bucket),
        ]
        _run(cmd, cwd=repo, env=env)
    else:
        print(f"[skip] sweep.tsv already exists: {sweep_out / 'sweep.tsv'}", flush=True)

    print(f"[done] pipeline outputs: {diag_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
