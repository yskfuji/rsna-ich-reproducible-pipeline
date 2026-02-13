"""Aggregate evaluation outputs under results/* into a single report.

This script is intentionally standalone and reads the JSON files produced by
`src.evaluation.evaluate_isles`.

Outputs:
- reports/latest_eval_report.json
- reports/latest_eval_report.md

Usage:
  /opt/anaconda3/envs/medseg_unet/bin/python reports/make_eval_report.py \
    --results-root results/3d_unet_medseg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvalRow:
    name: str
    path: str
    split: str | None
    model_path: str | None
    thresholds: list[float] | None
    min_size: int | None
    n: int | None
    n_gt_pos: int | None
    mean_dice: float | None
    median_dice: float | None
    max_dice: float | None
    min_dice: float | None
    n_dice_gt_0p1: int | None
    n_dice_gt_0p3: int | None
    detection_rate_case: float | None
    mean_fp_vox: float | None
    mean_fp_cc: float | None
    mean_fp_cc_vox: float | None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float(0.5 * (xs[mid - 1] + xs[mid]))


def load_eval_dir(dir_path: Path) -> EvalRow | None:
    summary_path = dir_path / "summary.json"
    metrics_path = dir_path / "metrics.json"
    if not summary_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return None

    dice_list: list[float] = []
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            for row in metrics:
                v = row.get("dice")
                if v is None:
                    continue
                try:
                    dice_list.append(float(v))
                except Exception:
                    continue
        except Exception:
            dice_list = []

    if dice_list:
        mean_dice = float(sum(dice_list) / len(dice_list))
        median_dice = _median(dice_list)
        max_dice = float(max(dice_list))
        min_dice = float(min(dice_list))
        n_gt_0p1 = int(sum(1 for d in dice_list if d > 0.1))
        n_gt_0p3 = int(sum(1 for d in dice_list if d > 0.3))
    else:
        mean_dice = median_dice = max_dice = min_dice = None
        n_gt_0p1 = n_gt_0p3 = None

    thresholds = summary.get("thresholds")
    thr_list: list[float] | None
    if isinstance(thresholds, list):
        thr_list = []
        for t in thresholds:
            ft = _safe_float(t)
            if ft is not None:
                thr_list.append(ft)
        if not thr_list:
            thr_list = None
    else:
        thr_list = None

    return EvalRow(
        name=dir_path.name,
        path=str(dir_path),
        split=summary.get("split"),
        model_path=summary.get("model_path"),
        thresholds=thr_list,
        min_size=_safe_int(summary.get("min_size")),
        n=_safe_int(summary.get("n")),
        n_gt_pos=_safe_int(summary.get("n_gt_pos")),
        mean_dice=mean_dice,
        median_dice=median_dice,
        max_dice=max_dice,
        min_dice=min_dice,
        n_dice_gt_0p1=n_gt_0p1,
        n_dice_gt_0p3=n_gt_0p3,
        detection_rate_case=_safe_float(summary.get("detection_rate_case")),
        mean_fp_vox=_safe_float(summary.get("mean_fp_vox")),
        mean_fp_cc=_safe_float(summary.get("mean_fp_cc")),
        mean_fp_cc_vox=_safe_float(summary.get("mean_fp_cc_vox")),
    )


def format_float(x: float | None, ndigits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{ndigits}f}"


def format_int(x: int | None) -> str:
    if x is None:
        return "-"
    return str(int(x))


def to_markdown(rows: list[EvalRow], results_root: Path) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append(f"# Evaluation report\n\nGenerated: {now}\n\nResults root: `{results_root}`\n")

    lines.append("## Summary\n")
    lines.append(
        "| name | split | thr | cc_min | n | mean_dice | median_dice | det_rate | mean_fp_vox | mean_fp_cc | |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for r in rows:
        thr = "-"
        if r.thresholds:
            thr = ",".join(f"{t:g}" for t in r.thresholds)
        rel = str(Path(r.path).resolve().relative_to(results_root.resolve())) if Path(r.path).resolve().is_relative_to(results_root.resolve()) else r.name
        lines.append(
            "| "
            + " | ".join(
                [
                    rel,
                    (r.split or "-"),
                    thr,
                    format_int(r.min_size),
                    format_int(r.n),
                    format_float(r.mean_dice),
                    format_float(r.median_dice),
                    format_float(r.detection_rate_case, ndigits=3),
                    format_float(r.mean_fp_vox, ndigits=2),
                    format_float(r.mean_fp_cc, ndigits=2),
                    "",
                ]
            )
            + "|"
        )

    lines.append("\n## Details\n")
    for r in rows:
        lines.append(f"### {r.name}\n")
        lines.append(f"- path: `{r.path}`")
        if r.model_path:
            lines.append(f"- model_path: `{r.model_path}`")
        if r.thresholds:
            lines.append(f"- thresholds: {', '.join(str(t) for t in r.thresholds)}")
        if r.min_size is not None:
            lines.append(f"- min_size: {r.min_size}")
        if r.n is not None:
            lines.append(f"- n: {r.n} (gt_pos={format_int(r.n_gt_pos)})")
        if r.mean_dice is not None:
            lines.append(
                "- dice: mean="
                + format_float(r.mean_dice)
                + " median="
                + format_float(r.median_dice)
                + " min="
                + format_float(r.min_dice)
                + " max="
                + format_float(r.max_dice)
                + f" (n>0.1={format_int(r.n_dice_gt_0p1)}, n>0.3={format_int(r.n_dice_gt_0p3)})"
            )
        if r.detection_rate_case is not None:
            lines.append(f"- detection_rate_case: {format_float(r.detection_rate_case, ndigits=3)}")
        if r.mean_fp_vox is not None:
            lines.append(
                "- FP: mean_fp_vox="
                + format_float(r.mean_fp_vox, ndigits=2)
                + " mean_fp_cc="
                + format_float(r.mean_fp_cc, ndigits=2)
                + " mean_fp_cc_vox="
                + format_float(r.mean_fp_cc_vox, ndigits=2)
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/3d_unet_medseg",
        help="Directory containing evaluation outputs (each subdir should have summary.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports",
        help="Output directory (default: reports)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(results_root.glob("**/summary.json"))
    rows: list[EvalRow] = []
    for sp in summary_paths:
        r = load_eval_dir(sp.parent)
        if r is not None:
            rows.append(r)

    # Sort primarily by split then mean_dice (desc).
    def _sort_key(x: EvalRow):
        split = x.split or ""
        md = x.mean_dice if x.mean_dice is not None else -1.0
        return (split, -md, x.name)

    rows = sorted(rows, key=_sort_key)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "rows": [asdict(r) for r in rows],
    }

    (out_dir / "latest_eval_report.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "latest_eval_report.md").write_text(to_markdown(rows, results_root))

    print(f"Wrote: {out_dir / 'latest_eval_report.md'}")
    print(f"Wrote: {out_dir / 'latest_eval_report.json'}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
