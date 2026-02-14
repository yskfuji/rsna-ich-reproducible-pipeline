"""Run fixedzscore model evaluation chain:

1) save-probs for the fixedzscore model
2) postprocess sweep for the fixedzscore probs (single)
3) ensemble average: (patch + phase1 + fixedzscore) probs
4) postprocess sweep for the ensemble probs

Designed for unstable terminals: run via nohup.

Example:
  /opt/anaconda3/bin/python tools/run_fixedzscore_single_and_ensemble.py \
    --repo /Users/yusukefujinami/ToReBrain/ToReBrain-pipeline \
    --fixed-config configs/generated/_plan_20251225/medseg_3d_unet_e20_dwi_adc_flair_phase2_patch562424_pv2bg_ffg02_bgdist12_tversky_ohem_fgccinv_a1_norm_fixedzscore.yaml \
    --patch-probs-dir results/diag/ensemble_pipeline_20251225_091332/saveprobs_medseg_3d_unet_e20_dwi_adc_flair_phase2_patch562424_pv2bg_ffg02_bgdist12_tversky_ohem_fgccinv_a1/probs \
    --phase1-probs-dir results/diag/ensemble_strong_patch_phase1_20251225_114153/saveprobs_medseg_3d_unet_e20_dwi_adc_flair_phase1_pv2bg_ffg02_bgdist12_tversky_ohem_fgccinv_a1/probs
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


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=None, help="Python executable (defaults to sys.executable-like behavior via PATH)")
    ap.add_argument("--repo", required=True, help="ToReBrain-pipeline root")
    ap.add_argument("--fixed-config", required=True, help="Training YAML config for fixedzscore model")
    ap.add_argument("--patch-probs-dir", required=True, help="Existing probs dir for strong patch model")
    ap.add_argument("--phase1-probs-dir", required=True, help="Existing probs dir for phase1 model")

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
    ap.add_argument("--sweep-min-sizes", default="0,20,40")
    ap.add_argument("--sweep-top-ks", default="0,1,2")
    ap.add_argument("--sweep-cc-scores", default="none")
    ap.add_argument("--sweep-cc-score-thrs", default="0.5")
    ap.add_argument("--focus-bucket", default="all")

    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    cfg_path = Path(args.fixed_config)
    if not cfg_path.is_absolute():
        cfg_path = (repo / cfg_path).resolve()

    cfg = _load_yaml(cfg_path)
    exp = str(cfg.get("experiment_name"))
    norm = str((cfg.get("data", {}) or {}).get("normalize", "legacy_zscore"))

    best_pt = repo / "runs" / "3d_unet" / exp / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    py = args.python or "python"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    stamp = _timestamp()
    diag_root = repo / "results" / "diag" / f"fixedzscore_chain_{stamp}"
    diag_root.mkdir(parents=True, exist_ok=True)

    # 1) Save probs for fixedzscore model
    fixed_out = diag_root / f"saveprobs_{exp}"
    fixed_probs = fixed_out / "probs"
    if not (fixed_probs.exists() and any(fixed_probs.glob("*.npz"))):
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
            str(fixed_out),
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
    else:
        print(f"[skip] fixedzscore probs already exist: {fixed_probs}", flush=True)

    # 2) Sweep on fixed probs (single)
    single_sweep_out = diag_root / "pp_sweep_fixedzscore"
    if not (single_sweep_out / "sweep.tsv").exists():
        cmd = [
            py,
            "tools/postprocess_sweep.py",
            "--probs-dir",
            str(fixed_probs),
            "--csv-path",
            str(args.csv_path),
            "--root",
            str(args.root),
            "--split",
            str(args.split),
            "--out-root",
            str(single_sweep_out),
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
        print(f"[skip] fixedzscore sweep.tsv already exists: {single_sweep_out / 'sweep.tsv'}", flush=True)

    # 3) Ensemble average: patch + phase1 + fixedzscore
    patch_probs = Path(args.patch_probs_dir)
    if not patch_probs.is_absolute():
        patch_probs = (repo / patch_probs).resolve()

    phase1_probs = Path(args.phase1_probs_dir)
    if not phase1_probs.is_absolute():
        phase1_probs = (repo / phase1_probs).resolve()

    ens_dir = diag_root / "ensemble_strong_plus_fixed"
    ens_probs = ens_dir / "probs"

    if not (ens_probs.exists() and any(ens_probs.glob("*.npz"))):
        cmd = [
            py,
            "tools/ensemble_probmaps.py",
            "--probs-dirs",
            str(patch_probs),
            str(phase1_probs),
            str(fixed_probs),
            "--out-probs-dir",
            str(ens_probs),
            "--dtype",
            "float16",
        ]
        _run(cmd, cwd=repo, env=env)
    else:
        print(f"[skip] ensemble probs already exist: {ens_probs}", flush=True)

    # 4) Sweep on ensemble probs
    ens_sweep_out = diag_root / "pp_sweep_ensemble"
    if not (ens_sweep_out / "sweep.tsv").exists():
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
            str(ens_sweep_out),
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
        print(f"[skip] ensemble sweep.tsv already exists: {ens_sweep_out / 'sweep.tsv'}", flush=True)

    print(f"[done] outputs: {diag_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
