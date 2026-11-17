#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
        raise ValueError("empty list")
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


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    cp = subprocess.run(cmd, text=True, env=env)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed ({cp.returncode}): {' '.join(cmd)}")


def _run_capture_json(cmd: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed ({cp.returncode}): {' '.join(cmd)}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")
    try:
        return json.loads(cp.stdout)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from stdout: {e}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Sweep seeds: retrain with MC-Dropout enabled during training, then evaluate baseline vs retrained "
            "on the same holdout split per seed. Outputs per-seed results + summary stats."
        )
    )

    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)
    p.add_argument("--rsna-root", required=True, type=str)
    p.add_argument("--preprocessed-root", required=True, type=str)
    p.add_argument("--baseline-ckpt", required=True, type=str)

    p.add_argument("--arch", default="convnext_tiny", type=str)
    p.add_argument("--image-size", default=384, type=int)
    p.add_argument("--windows", default="40,80;80,200;600,2800", type=str)
    p.add_argument("--preprocess", default="gpt52", type=str)
    p.add_argument("--stack-slices", default=3, type=int)

    p.add_argument("--limit-images", default=8000, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)
    p.add_argument("--split-by", default="study", type=str)

    p.add_argument("--epochs", default=1, type=int)
    p.add_argument("--lr", default=5e-6, type=float)
    p.add_argument("--batch-size", default=6, type=int)
    p.add_argument("--num-workers", default=0, type=int)

    p.add_argument("--dropout-stage-p", default=0.2, type=float)
    p.add_argument("--dropout-head-p", default=0.2, type=float)

    p.add_argument("--mc-samples", default=30, type=int)
    p.add_argument("--mc-seed-mode", default="same", type=str, help="same|zero|fixed:<int>")
    p.add_argument("--fit-temperature", action="store_true")

    p.add_argument("--out-root", default="results/mc_dropout_retrain/sweep_e1_lim8k_val05", type=str)
    p.add_argument("--skip-train-if-exists", action="store_true", help="Skip training if best.pt already exists")

    ns = p.parse_args(argv)

    seeds = _parse_int_list(ns.seeds)

    repo_root = Path(__file__).resolve().parents[1]
    train_entry = repo_root / "train_rsna_cnn2d_classifier.py"
    eval_script = repo_root / "tools" / "eval_rsna_uncertainty.py"

    if not train_entry.exists():
        raise SystemExit(f"Missing training entry: {train_entry}")
    if not eval_script.exists():
        raise SystemExit(f"Missing eval script: {eval_script}")

    env = dict(os.environ)

    out_root = Path(str(ns.out_root)).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_ckpt = Path(str(ns.baseline_ckpt)).expanduser().resolve()
    if not baseline_ckpt.exists():
        raise SystemExit(f"Missing baseline ckpt: {baseline_ckpt}")

    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        if str(ns.mc_seed_mode).lower() == "same":
            mc_seed = int(seed)
        elif str(ns.mc_seed_mode).lower() == "zero":
            mc_seed = 0
        elif str(ns.mc_seed_mode).lower().startswith("fixed:"):
            mc_seed = int(str(ns.mc_seed_mode).split(":", 1)[1])
        else:
            raise SystemExit("mc_seed_mode must be same|zero|fixed:<int>")

        out_dir = out_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        trained_ckpt = out_dir / "best.pt"

        # Train (dropout enabled during training)
        if not (bool(ns.skip_train_if_exists) and trained_ckpt.exists()):
            train_cmd = [
                sys.executable,
                str(train_entry),
                "train",
                "--rsna-root",
                str(ns.rsna_root),
                "--preprocessed-root",
                str(ns.preprocessed_root),
                "--out-dir",
                str(out_dir),
                "--limit-images",
                str(int(ns.limit_images)),
                "--dedup-before-split",
                "--val-frac",
                str(float(ns.val_frac)),
                "--split-by",
                str(ns.split_by),
                "--seed",
                str(int(seed)),
                "--epochs",
                str(int(ns.epochs)),
                "--lr",
                str(float(ns.lr)),
                "--weight-decay",
                "0.0",
                "--image-size",
                str(int(ns.image_size)),
                "--windows",
                str(ns.windows),
                "--preprocess",
                str(ns.preprocess),
                "--stack-slices",
                str(int(ns.stack_slices)),
                "--batch-size",
                str(int(ns.batch_size)),
                "--num-workers",
                str(int(ns.num_workers)),
                "--arch",
                str(ns.arch),
                "--pretrained",
                "--first-conv-init",
                "mean",
                "--input-normalize",
                "none",
                "--no-aug",
                "--no-scheduler",
                "--no-use-pos-weight",
                "--no-use-sampler",
                "--optimize-plain-loss",
                "--init-from",
                str(baseline_ckpt),
                "--dropout-stage-p",
                str(float(ns.dropout_stage_p)),
                "--dropout-head-p",
                str(float(ns.dropout_head_p)),
            ]
            _run(train_cmd, env=env)

        if not trained_ckpt.exists():
            raise SystemExit(f"Missing trained ckpt for seed {seed}: {trained_ckpt}")

        common_eval = [
            sys.executable,
            str(eval_script),
            "--rsna-root",
            str(ns.rsna_root),
            "--preprocessed-root",
            str(ns.preprocessed_root),
            "--arch",
            str(ns.arch),
            "--image-size",
            str(int(ns.image_size)),
            "--windows",
            str(ns.windows),
            "--preprocess",
            str(ns.preprocess),
            "--stack-slices",
            str(int(ns.stack_slices)),
            "--limit-images",
            str(int(ns.limit_images)),
            "--dedup-before-split",
            "--val-frac",
            str(float(ns.val_frac)),
            "--split-by",
            str(ns.split_by),
            "--seed",
            str(int(seed)),
            "--mc-samples",
            str(int(ns.mc_samples)),
            "--mc-seed",
            str(int(mc_seed)),
            "--dropout-stage-p",
            str(float(ns.dropout_stage_p)),
            "--dropout-head-p",
            str(float(ns.dropout_head_p)),
            "--out-curve-png",
            str(out_dir / "coverage_risk.png"),
            "--out-reliability-png",
            str(out_dir / "reliability_any.png"),
        ]
        if bool(ns.fit_temperature):
            common_eval.append("--fit-temperature")

        base = _run_capture_json(common_eval + ["--ckpt", str(baseline_ckpt)], env=env)
        retr = _run_capture_json(common_eval + ["--ckpt", str(trained_ckpt)], env=env)

        key_metrics = [
            "ece_any",
            "brier_any",
            "nll_weighted_logloss",
            "auroc_uncertainty_detect_error_any",
            "aurc_weighted_logloss",
            "accuracy_any_improve_pp",
            "temperature",
        ]
        delta: dict[str, float] = {}
        for k in key_metrics:
            a = base.get(k)
            b = retr.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                delta[k] = float(b) - float(a)

        per_seed.append(
            {
                "seed": int(seed),
                "mc_seed": int(mc_seed),
                "out_dir": str(out_dir),
                "trained_ckpt": str(trained_ckpt),
                "baseline": base,
                "retrained_with_dropout": retr,
                "delta_retrained_minus_baseline": delta,
            }
        )

        (out_dir / "compare.json").write_text(
            json.dumps(per_seed[-1], ensure_ascii=False, indent=2)
        )

    # Summaries
    summary: dict[str, Any] = {
        "seeds": list(seeds),
        "n_runs": int(len(per_seed)),
        "out_root": str(out_root),
        "per_seed": per_seed,
        "summary": {"baseline": {}, "retrained_with_dropout": {}, "delta": {}},
    }

    def _collect(which: str, key: str) -> list[float]:
        xs: list[float] = []
        for r in per_seed:
            src = r.get(which, {}) if which != "delta" else r.get("delta_retrained_minus_baseline", {})
            v = src.get(key) if isinstance(src, dict) else None
            if isinstance(v, (int, float)):
                xs.append(float(v))
        return xs

    metrics = [
        "ece_any",
        "brier_any",
        "nll_weighted_logloss",
        "auroc_uncertainty_detect_error_any",
        "aurc_weighted_logloss",
        "accuracy_any_improve_pp",
        "temperature",
    ]

    for m in metrics:
        summary["summary"]["baseline"][m] = _nan_stats(_collect("baseline", m))
        summary["summary"]["retrained_with_dropout"][m] = _nan_stats(_collect("retrained_with_dropout", m))
        summary["summary"]["delta"][m] = _nan_stats(_collect("delta", m))

    (out_root / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
