from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _parse_csv_list(s: str) -> list[str]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return parts


def _best_row(summary: dict, bucket: str) -> dict:
    per = summary.get("per_threshold") or []
    if not per:
        raise ValueError("summary.json has empty per_threshold")

    if bucket == "all":
        return max(per, key=lambda r: float(r.get("mean_dice") or -1.0))

    def bval(r: dict) -> float:
        by = (r.get("by_slice_spacing") or {}).get(bucket) or {}
        return float(by.get("mean_dice") or -1.0)

    return max(per, key=bval)


def _fmt_row(row: dict) -> str:
    def b(bucket: str) -> dict:
        return (row.get("by_slice_spacing") or {}).get(bucket) or {}

    le = b("le_3mm")
    gt = b("gt_3mm")
    thr = row.get("threshold")
    return (
        f"thr={thr} global={row.get('mean_dice')} "
        f"le_3mm={le.get('mean_dice')} det_le={le.get('detection_rate_case')} "
        f"gt_3mm={gt.get('mean_dice')}"
    )


def _row_to_dict(row: dict) -> dict:
    def b(bucket: str) -> dict:
        return (row.get("by_slice_spacing") or {}).get(bucket) or {}

    le = b("le_3mm")
    gt = b("gt_3mm")
    return {
        "threshold": float(row.get("threshold")),
        "mean_dice": float(row.get("mean_dice") or 0.0),
        "by_slice_spacing": {
            "le_3mm": {
                "mean_dice": float(le.get("mean_dice") or 0.0) if le else None,
                "detection_rate_case": float(le.get("detection_rate_case") or 0.0) if le else None,
            },
            "gt_3mm": {
                "mean_dice": float(gt.get("mean_dice") or 0.0) if gt else None,
                "detection_rate_case": float(gt.get("detection_rate_case") or 0.0) if gt else None,
            },
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Wait for best.pt then run evaluate_isles (tta=none/flip) and summarize.")
    p.add_argument("--exp-name", required=True, help="Experiment name under runs/3d_unet")
    p.add_argument("--runs-root", default="runs/3d_unet", help="Runs root (default: runs/3d_unet)")

    p.add_argument(
        "--expected-epochs",
        type=int,
        default=0,
        help="If >0, wait until val_threshold_last.json reports epoch>=expected_epochs (i.e., training finished).",
    )

    p.add_argument("--csv-path", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="test")

    p.add_argument("--patch-size", default="48,48,24")
    p.add_argument("--overlap", default="0.5")
    p.add_argument(
        "--thresholds",
        default="0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50",
    )
    p.add_argument("--min-size", default="0")
    p.add_argument("--top-k", default="0")
    p.add_argument("--cc-score", default="none")
    p.add_argument("--cc-score-thr", default="0.5")
    p.add_argument("--normalize", default="nonzero_zscore")
    p.add_argument("--resample-max-zoom-mm", default="2.0")
    p.add_argument("--slice-spacing-source", default="raw")

    p.add_argument("--ttas", default="none,flip", help="Comma-separated TTAs (none|flip|full)")
    p.add_argument("--out-root", default="results/diag")

    p.add_argument("--poll-sec", type=float, default=30.0)
    p.add_argument("--timeout-min", type=float, default=0.0, help="0 means no timeout")

    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (repo_root / runs_root).resolve()

    exp_dir = runs_root / args.exp_name
    best_path = exp_dir / "best.pt"
    last_meta_path = exp_dir / "val_threshold_last.json"

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()

    if args.expected_epochs and args.expected_epochs > 0:
        print(
            f"[wait] training completion: {last_meta_path} epoch>={args.expected_epochs} (then use best.pt)",
            flush=True,
        )
    else:
        print(f"[wait] best.pt: {best_path}", flush=True)
    t0 = time.time()
    while True:
        if args.expected_epochs and args.expected_epochs > 0:
            if last_meta_path.exists() and last_meta_path.stat().st_size > 0:
                try:
                    meta = json.loads(last_meta_path.read_text())
                    ep = int(meta.get("epoch") or 0)
                except Exception:
                    ep = 0
                if ep >= int(args.expected_epochs):
                    if best_path.exists() and best_path.stat().st_size > 0:
                        break
        else:
            if best_path.exists() and best_path.stat().st_size > 0:
                break
        if args.timeout_min and (time.time() - t0) > (float(args.timeout_min) * 60.0):
            raise TimeoutError(f"Timed out waiting for: {best_path}")
        time.sleep(float(args.poll_sec))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ttas = [t.strip().lower() for t in _parse_csv_list(args.ttas)]
    summaries: dict[str, Path] = {}
    best_by_tta: dict[str, dict] = {}

    for tta in ttas:
        out_dir = out_root / f"test_eval_{tta}_{args.exp_name}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_isles",
            "--model-path",
            str(best_path),
            "--csv-path",
            str(Path(args.csv_path).expanduser()),
            "--root",
            str(Path(args.root).expanduser()),
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
            str(args.min_size),
            "--top-k",
            str(args.top_k),
            "--cc-score",
            str(args.cc_score),
            "--cc-score-thr",
            str(args.cc_score_thr),
            "--normalize",
            str(args.normalize),
            "--tta",
            str(tta),
            "--resample-max-zoom-mm",
            str(args.resample_max_zoom_mm),
            "--slice-spacing-source",
            str(args.slice_spacing_source),
            "--quiet",
        ]

        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root)

        print(f"[eval] tta={tta} -> {out_dir}", flush=True)
        subprocess.run(cmd, check=True, env=env)

        summary_path = out_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json: {summary_path}")
        summaries[tta] = summary_path

        s = json.loads(summary_path.read_text())
        r_all = _best_row(s, "all")
        r_le = _best_row(s, "le_3mm")
        best_by_tta[tta] = {
            "eval_config": {
                "exp_name": str(args.exp_name),
                "split": str(args.split),
                "patch_size": str(args.patch_size),
                "overlap": float(args.overlap),
                "thresholds": str(args.thresholds),
                "normalize": str(args.normalize),
                "tta": str(tta),
                "resample_max_zoom_mm": float(args.resample_max_zoom_mm),
                "slice_spacing_source": str(args.slice_spacing_source),
                "postprocess": {
                    "min_size": int(float(args.min_size)),
                    "top_k": int(float(args.top_k)),
                    "cc_score": str(args.cc_score),
                    "cc_score_thr": float(args.cc_score_thr),
                },
            },
            "best_all": _row_to_dict(r_all),
            "best_le_3mm": _row_to_dict(r_le),
        }
        (out_dir / "best_thresholds.json").write_text(json.dumps(best_by_tta[tta], indent=2) + "\n")

    print("\n=== summary (best global / best le_3mm) ===", flush=True)
    for tta, sp in summaries.items():
        s = json.loads(sp.read_text())
        r_all = _best_row(s, "all")
        r_le = _best_row(s, "le_3mm")
        print(f"[{tta}] best_all  : {_fmt_row(r_all)}")
        print(f"[{tta}] best_le_3: {_fmt_row(r_le)}")

    # One file that points to all TTAs (useful when running --ttas none,flip)
    # The per-TTA directory also contains its own best_thresholds.json.
    out_parent = out_root / f"test_eval_{args.exp_name}_{stamp}"
    out_parent.mkdir(parents=True, exist_ok=True)
    (out_parent / "best_thresholds_by_tta.json").write_text(json.dumps(best_by_tta, indent=2) + "\n")


if __name__ == "__main__":
    main()
