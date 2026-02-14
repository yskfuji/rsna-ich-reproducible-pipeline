"""Wait for k-fold trainings to finish, then run eval (threshold sweep) per fold and prob-average ensemble.

Designed for unstable terminals: run it with log redirection.

Workflow:
1) For each fold config YAML:
   - wait until runs/3d_unet/<experiment_name>/best.pt exists
   - optionally wait until val_threshold_last.json epoch>=train.epochs
   - run `python -m src.evaluation.evaluate_isles --save-probs` to produce out_dir/probs + summary.json
2) Average probs across folds with tools/ensemble_probmaps.py
3) Evaluate the ensemble via `python -m src.evaluation.evaluate_isles --probs-dir ...` (threshold sweep)

Outputs:
- results/diag/kfold_ensemble_<STAMP>/
  - saveprobs_<exp>/ (per fold)
  - ensemble/probs/ (averaged NPZ)
  - eval_ensemble_<split>/ (summary.json)
  - summary_compact.json (best threshold per fold + ensemble)

Notes:
- This script intentionally does *only* threshold sweep (no postprocess grid).
- For prob-averaging, all fold runs must be evaluated on the same case list.
    That means: the provided --csv-path and --split must select identical case_ids across folds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Optional, cast

import yaml


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_yaml(p: Path) -> dict[str, Any]:
    obj = yaml.safe_load(p.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping, got: {type(obj).__name__}")
    return cast(dict[str, Any], obj)


def _wait_for_best(
    *,
    cfg_path: Path,
    repo: Path,
    wait_complete: bool,
    poll_s: float,
) -> None:
    last_msg: str | None = None
    while True:
        cfg = _load_yaml(cfg_path)
        exp = str(cfg.get("experiment_name") or cfg_path.stem)
        train_cfg = cast(dict[str, Any], cfg.get("train") or {})
        epochs = int(train_cfg.get("epochs", 0) or 0)

        exp_dir = repo / "runs" / "3d_unet" / exp
        best_pt = exp_dir / "best.pt"
        last_meta = exp_dir / "val_threshold_last.json"
        expected_epochs = epochs if bool(wait_complete) else 0

        if expected_epochs > 0:
            msg = f"[wait] training completion: {last_meta} epoch>={expected_epochs} (and best.pt)"
        else:
            msg = f"[wait] best.pt: {best_pt}"
        if msg != last_msg:
            print(msg, flush=True)
            last_msg = msg

        if expected_epochs > 0:
            if last_meta.exists() and last_meta.stat().st_size > 0:
                try:
                    meta = json.loads(last_meta.read_text())
                    ep = int(meta.get("epoch") or 0)
                except Exception:
                    ep = 0
                if ep >= expected_epochs and best_pt.exists() and best_pt.stat().st_size > 0:
                    return
        else:
            if best_pt.exists() and best_pt.stat().st_size > 0:
                return

        time.sleep(poll_s)


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _best_row(summary: dict[str, Any], *, focus_bucket: str = "all") -> dict[str, Any]:
    per_any: Any = summary.get("per_threshold") or []
    per = cast(list[dict[str, Any]], per_any)
    if not per:
        raise ValueError("summary.json has empty per_threshold")

    fb = str(focus_bucket).strip() or "all"
    if fb in {"all", ""}:
        return max(per, key=lambda r: float(r.get("mean_dice") or -1.0))

    def score(r: dict[str, Any]) -> float:
        by = cast(dict[str, Any], r.get("by_slice_spacing") or {})
        sub = cast(dict[str, Any], by.get(fb) or {})
        v = sub.get("mean_dice")
        return float(v) if v is not None else -1.0

    return max(per, key=score)


def _extract_fold(exp_name: str) -> Optional[int]:
    m = re.search(r"(?:^|_)f(\d+)(?:_|$)", exp_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _discover_configs(configs: list[str], configs_dir: str) -> list[Path]:
    out: list[Path] = []
    if configs:
        out = [Path(p).expanduser() for p in configs]
    else:
        d = Path(configs_dir).expanduser()
        if not d.exists():
            raise FileNotFoundError(str(d))
        out = sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml"))

    if not out:
        raise ValueError("No config YAMLs provided/found")
    return out


def _case_ids_in_probs_dir(d: Path) -> set[str]:
    return {p.stem for p in d.glob("*.npz")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True, help="Python executable")
    ap.add_argument("--repo", required=True, help="ToReBrain-pipeline root")
    ap.add_argument("--configs", nargs="*", default=[], help="Fold training YAML configs")
    ap.add_argument("--configs-dir", default="", help="Directory containing fold YAMLs (used if --configs empty)")

    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")

    ap.add_argument("--patch-size", default="96,96,96")
    ap.add_argument("--overlap", default="0.5")
    ap.add_argument(
        "--thresholds",
        default=(
            "0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.80"
        ),
    )
    ap.add_argument("--tta", default="none")
    ap.add_argument("--resample-max-zoom-mm", default="0.0")
    ap.add_argument("--slice-spacing-source", default="effective")

    ap.add_argument(
        "--wait-complete",
        action="store_true",
        help="Wait until val_threshold_last.json epoch>=train.epochs (instead of just best.pt).",
    )
    ap.add_argument("--poll-seconds", type=float, default=120.0)

    ap.add_argument(
        "--focus-bucket",
        default="all",
        help="Which bucket to use when selecting best threshold: all|le_3mm|gt_3mm...",
    )

    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    py = str(Path(args.python).expanduser().resolve())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    cfg_paths = _discover_configs(list(args.configs), str(args.configs_dir))
    cfg_paths = [p if p.is_absolute() else (repo / p).resolve() for p in cfg_paths]

    stamp = _timestamp()
    diag_root = repo / "results" / "diag" / f"kfold_ensemble_{stamp}"
    diag_root.mkdir(parents=True, exist_ok=True)

    probs_dirs: list[Path] = []
    fold_rows: list[dict[str, Any]] = []

    for cfg_path in cfg_paths:
        _wait_for_best(cfg_path=cfg_path, repo=repo, wait_complete=bool(args.wait_complete), poll_s=float(args.poll_seconds))

        cfg = _load_yaml(cfg_path)
        exp = str(cfg.get("experiment_name") or cfg_path.stem)
        data_cfg = cast(dict[str, Any], cfg.get("data") or {})
        norm = str(data_cfg.get("normalize", "nonzero_zscore"))
        fold = _extract_fold(exp)

        exp_dir = repo / "runs" / "3d_unet" / exp
        best_pt = exp_dir / "best.pt"

        out_dir = diag_root / f"saveprobs_{exp}"
        probs_out = out_dir / "probs"

        if probs_out.exists() and any(probs_out.glob("*.npz")) and (out_dir / "summary.json").exists():
            print(f"[skip] fold probs+summary already exist: {out_dir}", flush=True)
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

        s = json.loads((out_dir / "summary.json").read_text())
        best = _best_row(s, focus_bucket=str(args.focus_bucket))
        fold_rows.append(
            {
                "fold": fold,
                "experiment_name": exp,
                "summary_dir": str(out_dir),
                "best_threshold": float(best.get("threshold") or 0.0),
                "best_mean_dice": float(best.get("mean_dice") or 0.0),
            }
        )

    # Sanity check: prob-averaging requires identical case_id sets across dirs.
    case_sets = [_case_ids_in_probs_dir(d) for d in probs_dirs]
    if not case_sets or any(len(s) == 0 for s in case_sets):
        raise RuntimeError("Empty probs directory detected; cannot ensemble")
    base_set = case_sets[0]
    mismatch = [i for i, s in enumerate(case_sets) if s != base_set]
    if mismatch:
        details: dict[str, Any] = {
            "n_dirs": len(probs_dirs),
            "base_n_cases": len(base_set),
            "mismatch_indices": mismatch,
            "example_only_in_base": sorted(list(base_set - case_sets[mismatch[0]]))[:10],
            "example_only_in_mismatch": sorted(list(case_sets[mismatch[0]] - base_set))[:10],
        }
        raise RuntimeError(
            "Cannot prob-average folds because case_id sets differ across probs dirs. "
            "Use an eval --csv-path/--split that selects the same cases for all folds (e.g., fixed test), "
            "or re-run with a shared validation split. Details: " + json.dumps(details)
        )

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

    ens_eval_dir = diag_root / f"eval_ensemble_{str(args.split)}"
    if not (ens_eval_dir / "summary.json").exists():
        cmd = [
            py,
            "-m",
            "src.evaluation.evaluate_isles",
            "--probs-dir",
            str(ens_probs),
            "--csv-path",
            str(args.csv_path),
            "--root",
            str(args.root),
            "--split",
            str(args.split),
            "--out-dir",
            str(ens_eval_dir),
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
            "nonzero_zscore",
            "--tta",
            "none",
            "--resample-max-zoom-mm",
            str(args.resample_max_zoom_mm),
            "--slice-spacing-source",
            str(args.slice_spacing_source),
            "--quiet",
        ]
        _run(cmd, cwd=repo, env=env)

    s_ens = json.loads((ens_eval_dir / "summary.json").read_text())
    best_ens = _best_row(s_ens, focus_bucket=str(args.focus_bucket))

    out_obj: dict[str, Any] = {
        "diag_root": str(diag_root),
        "split": str(args.split),
        "focus_bucket": str(args.focus_bucket),
        "thresholds": str(args.thresholds),
        "per_fold": sorted(
            fold_rows,
            key=lambda r: (r["fold"] is None, int(r["fold"]) if r["fold"] is not None else 9999, r["experiment_name"]),
        ),
        "ensemble": {
            "probs_dir": str(ens_probs),
            "eval_dir": str(ens_eval_dir),
            "best_threshold": float(best_ens.get("threshold") or 0.0),
            "best_mean_dice": float(best_ens.get("mean_dice") or 0.0),
        },
    }
    (diag_root / "summary_compact.json").write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n")

    print(f"[done] outputs: {diag_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
