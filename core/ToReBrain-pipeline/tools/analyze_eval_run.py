#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _best_threshold(summary: dict[str, Any]) -> float:
    per = summary.get("per_threshold")
    if not isinstance(per, list) or not per:
        raise ValueError("summary.json missing per_threshold")
    best = max(per, key=lambda r: float((r or {}).get("mean_dice") or 0.0))
    return float(best["threshold"])


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = int(round((len(xs) - 1) * q))
    return float(xs[i])


def _extract_dice_at_best(row: dict[str, Any], best_thr: float) -> float:
    key = f"dice@{best_thr:g}"
    v = row.get(key)
    if v is None:
        v = row.get("dice")
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def _classify_case(*, detected: Any, precision: float, recall: float) -> str:
    if detected is False:
        return "missed"
    # Heuristic buckets to quickly separate dominant error modes.
    if recall < 0.35 and precision >= 0.35:
        return "FN-dominant"
    if precision < 0.35 and recall >= 0.35:
        return "FP-dominant"
    if precision < 0.35 and recall < 0.35:
        return "both-bad"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a single eval dir containing summary.json + metrics.json")
    ap.add_argument("--eval-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary_p = eval_dir / "summary.json"
    metrics_p = eval_dir / "metrics.json"
    if not summary_p.exists():
        raise FileNotFoundError(f"not found: {summary_p}")
    if not metrics_p.exists():
        raise FileNotFoundError(f"not found: {metrics_p}")

    summary = _read_json(summary_p)
    rows = _read_json(metrics_p)
    if not isinstance(rows, list):
        raise ValueError("metrics.json must be a list")

    best_thr = _best_threshold(summary)

    cases: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = r.get("case_id")
        if not cid:
            continue
        dice_best = _extract_dice_at_best(r, best_thr)
        tp = _safe_int(r.get("tp_vox"))
        fp = _safe_int(r.get("fp_vox"))
        fn = _safe_int(r.get("fn_vox"))
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        tag = _classify_case(detected=r.get("detected"), precision=precision, recall=recall)
        cases.append(
            {
                "case_id": str(cid),
                "dice_best": float(dice_best),
                "detected": bool(r.get("detected")) if r.get("detected") is not None else None,
                "gt_vox": _safe_float(r.get("gt_vox")),
                "pred_vox": _safe_float(r.get("pred_vox")),
                "tp_vox": float(tp),
                "fp_vox": float(fp),
                "fn_vox": float(fn),
                "precision": float(precision),
                "recall": float(recall),
                "error_tag": tag,
                "fp_cc": _safe_float(r.get("fp_cc")),
                "gt_size_bucket": r.get("gt_size_bucket"),
                "slice_spacing_bucket": r.get("slice_spacing_bucket"),
            }
        )

    dices = [c["dice_best"] for c in cases]

    by_size: dict[str, list[float]] = {}
    for c in cases:
        k = c.get("gt_size_bucket")
        if isinstance(k, str) and k:
            by_size.setdefault(k, []).append(float(c["dice_best"]))

    missed = [c for c in cases if c.get("detected") is False]
    worst = sorted(cases, key=lambda c: float(c["dice_best"]))[: max(0, int(args.top_k))]
    cases_sorted = sorted(cases, key=lambda c: float(c["dice_best"]))

    report = {
        "eval_dir": str(eval_dir),
        "split": summary.get("split"),
        "n": len(cases),
        "best_threshold": best_thr,
        "dice": {
            "mean": _mean(dices),
            "median": _percentile(dices, 0.50),
            "p10": _percentile(dices, 0.10),
            "p25": _percentile(dices, 0.25),
            "p75": _percentile(dices, 0.75),
            "p90": _percentile(dices, 0.90),
        },
        "detection_rate_case": summary.get("detection_rate_case"),
        "by_gt_size_bucket": {k: {"n": len(v), "mean": _mean(v), "median": _percentile(v, 0.50)} for k, v in sorted(by_size.items())},
        "missed_cases": [c["case_id"] for c in missed],
        "worst_cases": [{"case_id": c["case_id"], "dice_best": c["dice_best"], "gt_vox": c["gt_vox"], "pred_vox": c["pred_vox"], "fp_vox": c["fp_vox"], "fn_vox": c["fn_vox"], "gt_size_bucket": c["gt_size_bucket"]} for c in worst],
    }

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "analysis_report.json").write_text(json.dumps(report, indent=2))

    # CSV for quick scanning
    with (out_dir / "worst_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "dice_best",
                "detected",
                "gt_vox",
                "pred_vox",
                "tp_vox",
                "fp_vox",
                "fn_vox",
                "precision",
                "recall",
                "error_tag",
                "fp_cc",
                "gt_size_bucket",
                "slice_spacing_bucket",
            ],
        )
        w.writeheader()
        for c in worst:
            w.writerow(c)

    with (out_dir / "cases_sorted.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "dice_best",
                "detected",
                "gt_vox",
                "pred_vox",
                "tp_vox",
                "fp_vox",
                "fn_vox",
                "precision",
                "recall",
                "error_tag",
                "fp_cc",
                "gt_size_bucket",
                "slice_spacing_bucket",
            ],
        )
        w.writeheader()
        for c in cases_sorted:
            w.writerow(c)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
