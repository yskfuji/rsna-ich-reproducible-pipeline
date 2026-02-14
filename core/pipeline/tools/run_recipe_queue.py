"""Run a queue of recipe trainings, then run evaluate_isles on val/test with best_all threshold saved.

Why:
- VS Code terminal sessions may disconnect.
- We want one background process to run: train -> evaluate (threshold sweep) -> persist best_all threshold.

Usage example:
  /opt/anaconda3/envs/medseg_unet/bin/python tools/run_recipe_queue.py \
    --python /opt/anaconda3/envs/medseg_unet/bin/python \
    --repo /Users/yusukefujinami/ToReBrain/ToReBrain-pipeline \
    --configs configs/generated/_recipe_20251227/a.yaml configs/generated/_recipe_20251227/b.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import json

import yaml


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _as_csv_ints(xs: list[int]) -> str:
    return ",".join(str(int(x)) for x in xs)


def _load_cfg(cfg_path: Path) -> dict:
    obj = yaml.safe_load(cfg_path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML (expected mapping): {cfg_path}")
    return obj


def _find_latest_best_thresholds(eval_out_root: Path, exp_name: str) -> Path | None:
    # Produced by tools/wait_train_and_eval.py: <out_root>/test_eval_<tta>_<exp>_<stamp>/best_thresholds.json
    patt = f"test_eval_*_{exp_name}_*"
    cands = [p for p in eval_out_root.glob(patt) if p.is_dir()]
    if not cands:
        return None
    cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    for d in cands:
        p = d / "best_thresholds.json"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]), help="Repo root (ToReBrain-pipeline)")
    ap.add_argument("--configs", nargs="+", required=True, help="List of YAML config paths")

    ap.add_argument("--eval-split", default="val", help="val|test")
    ap.add_argument("--eval-ttas", default="none", help="Comma-separated TTAs passed to wait_train_and_eval.py")
    ap.add_argument("--eval-overlap", default="0.5")
    ap.add_argument(
        "--eval-thresholds",
        default="0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50",
    )
    ap.add_argument("--eval-min-size", default="0", help="Connected-component min_size for evaluation (0 disables)")
    ap.add_argument("--eval-top-k", default="0", help="Keep top-k largest components for evaluation (0 disables)")
    ap.add_argument(
        "--eval-cc-score",
        default="none",
        help="Connected-component score filter for evaluation: none|max_prob|p95_prob|mean_prob",
    )
    ap.add_argument(
        "--eval-cc-score-thr",
        default="0.5",
        help="CC score threshold for evaluation (used when --eval-cc-score != none)",
    )
    ap.add_argument("--eval-resample-max-zoom-mm", default="2.0")
    ap.add_argument("--eval-slice-spacing-source", default="raw")
    ap.add_argument("--eval-out-root", default="results/diag/_recipe_20251227")

    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    py = str(Path(args.python).resolve())
    eval_out_root = Path(args.eval_out_root)
    if not eval_out_root.is_absolute():
        eval_out_root = (repo / eval_out_root).resolve()

    log_root = eval_out_root / "recipe_queue_logs" / f"queue_{_timestamp()}"
    log_root.mkdir(parents=True, exist_ok=True)

    results_jsonl = log_root / "results.jsonl"
    results_json = log_root / "results.json"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    print(f"[queue] repo={repo}")
    print(f"[queue] python={py}")
    print(f"[queue] eval_out_root={eval_out_root}")
    print(f"[queue] logs={log_root}")
    print(f"[queue] n_configs={len(args.configs)}")

    for idx, cfg in enumerate(args.configs, start=1):
        cfg_path = Path(cfg)
        if not cfg_path.is_absolute():
            cfg_path = (repo / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))

        cfg_obj = _load_cfg(cfg_path)
        exp_name = str(cfg_obj.get("experiment_name") or "").strip()
        if not exp_name:
            raise ValueError(f"Missing experiment_name in {cfg_path}")

        data = cfg_obj.get("data") or {}
        train = cfg_obj.get("train") or {}

        csv_path = str(data.get("csv_path") or "").strip()
        root = str(data.get("root") or "").strip()
        normalize = str(data.get("normalize") or "nonzero_zscore").strip()
        patch_size = data.get("patch_size")
        epochs = int(train.get("epochs") or 0)

        if not csv_path or not root:
            raise ValueError(f"Missing data.csv_path/root in {cfg_path}")
        if not isinstance(patch_size, list) or len(patch_size) != 3:
            raise ValueError(f"Invalid data.patch_size (expected 3 ints) in {cfg_path}")

        patch_size_csv = _as_csv_ints([int(x) for x in patch_size])

        log_path = log_root / f"{idx:02d}_{cfg_path.stem}.log"
        print(f"[queue] ({idx}/{len(args.configs)}) train {exp_name} ({cfg_path.name})")

        train_cmd = [py, "-m", "src.training.train_3d_unet", "--config", str(cfg_path)]
        with log_path.open("wb") as f:
            f.write(("[train_cmd] " + " ".join(train_cmd) + "\n").encode("utf-8"))
            f.flush()
            p = subprocess.Popen(
                train_cmd,
                cwd=str(repo),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            rc = p.wait()
            f.write((f"\n[train_exit] code={rc}\n").encode("utf-8"))
            f.flush()

        if rc != 0:
            print(f"[queue] FAILED train exp={exp_name} rc={rc} (see {log_path})")
            return rc

        print(f"[queue] ({idx}/{len(args.configs)}) eval split={args.eval_split} exp={exp_name}")

        eval_cmd = [
            py,
            str((repo / "tools" / "wait_train_and_eval.py").resolve()),
            "--exp-name",
            exp_name,
            "--expected-epochs",
            str(epochs),
            "--csv-path",
            csv_path,
            "--root",
            root,
            "--split",
            str(args.eval_split),
            "--patch-size",
            patch_size_csv,
            "--overlap",
            str(args.eval_overlap),
            "--thresholds",
            str(args.eval_thresholds),
            "--min-size",
            str(args.eval_min_size),
            "--top-k",
            str(args.eval_top_k),
            "--cc-score",
            str(args.eval_cc_score),
            "--cc-score-thr",
            str(args.eval_cc_score_thr),
            "--normalize",
            normalize,
            "--ttas",
            str(args.eval_ttas),
            "--resample-max-zoom-mm",
            str(args.eval_resample_max_zoom_mm),
            "--slice-spacing-source",
            str(args.eval_slice_spacing_source),
            "--out-root",
            str(eval_out_root),
        ]

        with log_path.open("ab") as f:
            f.write(("\n[eval_cmd] " + " ".join(eval_cmd) + "\n").encode("utf-8"))
            f.flush()
            p = subprocess.Popen(
                eval_cmd,
                cwd=str(repo),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            rc = p.wait()
            f.write((f"\n[eval_exit] code={rc}\n").encode("utf-8"))
            f.flush()

        if rc != 0:
            print(f"[queue] FAILED eval exp={exp_name} rc={rc} (see {log_path})")
            return rc

        best_path = _find_latest_best_thresholds(eval_out_root, exp_name)
        if best_path is None:
            print(f"[queue] WARN best_thresholds.json not found for exp={exp_name}")
        else:
            rec = {
                "timestamp": _timestamp(),
                "exp_name": exp_name,
                "config": str(cfg_path),
                "split": str(args.eval_split),
                "eval_postprocess": {
                    "min_size": int(float(args.eval_min_size)),
                    "top_k": int(float(args.eval_top_k)),
                    "cc_score": str(args.eval_cc_score),
                    "cc_score_thr": float(args.eval_cc_score_thr),
                },
                "best_thresholds_path": str(best_path),
                "best_thresholds": json.loads(best_path.read_text()),
            }
            with results_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # write a convenience aggregated JSON too
            all_recs: list[dict] = []
            for line in results_jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                all_recs.append(json.loads(line))
            results_json.write_text(json.dumps({"runs": all_recs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"[queue] done exp={exp_name} (log={log_path.name})")

    print("[queue] ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
