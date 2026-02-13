#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    cp = subprocess.run(cmd, text=True, env=env)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)


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
            "Train a MC-Dropout-enabled model (dropout active during training) and compare uncertainty metrics "
            "against a baseline checkpoint on the same holdout split."
        )
    )

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
    p.add_argument("--seed", default=0, type=int, help="Split/train seed")

    p.add_argument("--epochs", default=1, type=int)
    p.add_argument("--lr", default=5e-6, type=float)
    p.add_argument("--batch-size", default=6, type=int)
    p.add_argument("--num-workers", default=0, type=int)

    p.add_argument("--dropout-stage-p", default=0.2, type=float)
    p.add_argument("--dropout-head-p", default=0.2, type=float)

    p.add_argument(
        "--out-dir",
        default="results/mc_dropout_retrain/seed0",
        type=str,
        help="Training output dir (best.pt will be used)",
    )

    p.add_argument("--mc-samples", default=30, type=int)
    p.add_argument("--mc-seed", default=0, type=int)
    p.add_argument("--fit-temperature", action="store_true")

    ns = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    train_entry = repo_root / "train_rsna_cnn2d_classifier.py"
    eval_script = repo_root / "tools" / "eval_rsna_uncertainty.py"

    if not train_entry.exists():
        raise SystemExit(f"Missing training entry: {train_entry}")
    if not eval_script.exists():
        raise SystemExit(f"Missing eval script: {eval_script}")

    env = dict(os.environ)

    out_dir = Path(str(ns.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Train with dropout enabled (active in model.train())
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
        str(int(ns.seed)),
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
        str(Path(str(ns.baseline_ckpt)).expanduser().resolve()),
        "--dropout-stage-p",
        str(float(ns.dropout_stage_p)),
        "--dropout-head-p",
        str(float(ns.dropout_head_p)),
    ]

    _run(train_cmd, env=env)

    trained_ckpt = out_dir / "best.pt"
    if not trained_ckpt.exists():
        raise SystemExit(f"Training finished but missing checkpoint: {trained_ckpt}")

    # 2) Evaluate baseline and trained on the same holdout split (seed controls split)
    common_eval_args = [
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
        str(int(ns.seed)),
        "--mc-samples",
        str(int(ns.mc_samples)),
        "--mc-seed",
        str(int(ns.mc_seed)),
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
        common_eval_args.append("--fit-temperature")

    base = _run_capture_json([sys.executable] + common_eval_args + ["--ckpt", str(Path(str(ns.baseline_ckpt)).expanduser().resolve())], env=env)
    retr = _run_capture_json([sys.executable] + common_eval_args + ["--ckpt", str(trained_ckpt)], env=env)

    key_metrics = [
        "ece_any",
        "brier_any",
        "nll_weighted_logloss",
        "auroc_uncertainty_detect_error_any",
        "aurc_weighted_logloss",
        "accuracy_any_improve_pp",
        "temperature",
    ]

    delta: dict[str, Any] = {}
    for k in key_metrics:
        a = base.get(k)
        b = retr.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[k] = float(b) - float(a)

    out = {
        "seed": int(ns.seed),
        "baseline_ckpt": str(Path(str(ns.baseline_ckpt)).expanduser().resolve()),
        "trained_ckpt": str(trained_ckpt),
        "baseline": base,
        "retrained_with_dropout": retr,
        "delta_retrained_minus_baseline": delta,
        "out_dir": str(out_dir),
    }

    (out_dir / "compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
