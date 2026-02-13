#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _maybe_load_spacing_groups(summary: dict[str, Any]) -> dict[str, str]:
    """Best-effort per-case slice spacing group (e.g., le_3mm/gt_3mm).

    We prefer the same metadata path as evaluate_isles.py: sample['meta']['zooms_mm'].
    """
    csv_path = summary.get("csv_path")
    root = summary.get("root")
    split = summary.get("split")
    normalize = summary.get("normalize")
    ss_bins = summary.get("slice_spacing_bins_mm")
    if not (csv_path and root and split and ss_bins):
        return {}
    # `slice_spacing_bins_mm` can be stored as:
    # - scalar (e.g., 3.0)
    # - list/tuple (e.g., [3.0])
    # - string (e.g., "3.0" or "3.0,4.0")
    bins: list[float]
    try:
        if isinstance(ss_bins, (int, float)):
            bins = [float(ss_bins)]
        elif isinstance(ss_bins, str):
            parts = [p.strip() for p in ss_bins.replace(";", ",").split(",") if p.strip()]
            bins = [float(p) for p in parts]
        else:
            bins = [float(x) for x in ss_bins]
    except Exception:
        return {}

    try:
        # Import locally to keep this script runnable even without PYTHONPATH.
        from src.datasets.isles_dataset import IslesVolumeDataset  # type: ignore

        ds = IslesVolumeDataset(
            str(csv_path),
            split=str(split),
            root=str(root),
            transform=None,
            normalize=str(normalize) if normalize else "legacy_zscore",
            allow_missing_label=bool(summary.get("allow_missing_label") or False),
        )
    except Exception:
        return {}

    def bucket(mm: float | None) -> str:
        if mm is None:
            return "unknown"
        if not bins:
            return "unknown"
        if mm <= float(bins[0]):
            return f"le_{float(bins[0]):g}mm"
        return f"gt_{float(bins[0]):g}mm"

    out: dict[str, str] = {}
    for i in range(len(ds)):
        s = ds[i]
        cid = s.get("case_id")
        if not cid:
            continue
        meta = s.get("meta") or {}
        zooms = None
        if isinstance(meta, dict) and "zooms_mm" in meta:
            z = meta.get("zooms_mm")
            if isinstance(z, (list, tuple)) and len(z) >= 3:
                try:
                    zooms = [float(z[0]), float(z[1]), float(z[2])]
                except Exception:
                    zooms = None
        mm = None
        if zooms is not None:
            try:
                mm = float(max(zooms))
            except Exception:
                mm = None
        out[str(cid)] = bucket(mm)
    return out


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _best_threshold(summary: dict[str, Any]) -> float:
    best = max(summary["per_threshold"], key=lambda r: float(r.get("mean_dice") or 0.0))
    return float(best["threshold"])


def _extract_case_row(row: dict[str, Any], thr: float) -> dict[str, Any]:
    # metrics.json stores per-case base dice at whatever threshold was used for generating it,
    # plus optional dice@<thr> fields when multiple thresholds were evaluated.
    key = f"dice@{thr:g}"
    dice = row.get(key, row.get("dice"))

    out = {
        "case_id": row.get("case_id"),
        "dice": _safe_float(dice),
        "detected": row.get("detected"),
        "gt_vox": _safe_float(row.get("gt_vox")),
        "pred_vox": _safe_float(row.get("pred_vox")),
        "tp_vox": _safe_float(row.get("tp_vox")),
        "fp_vox": _safe_float(row.get("fp_vox")),
        "fn_vox": _safe_float(row.get("fn_vox")),
        "fp_cc": _safe_float(row.get("fp_cc")),
        "fp_cc_vox": _safe_float(row.get("fp_cc_vox")),
    }
    # some runs also add stratification keys; keep if present
    for k in ("slice_spacing_bin", "slice_spacing_group", "slice_spacing"):
        if k in row:
            out[k] = row.get(k)
    return out


def _load_run(run_dir: Path, split_hint: str | None = None) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    summary_p = run_dir / "summary.json"
    metrics_p = run_dir / "metrics.json"
    if not summary_p.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_p}")
    if not metrics_p.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_p}")

    summary = _read_json(summary_p)
    thr = _best_threshold(summary)
    split = str(summary.get("split") or split_hint or "")

    spacing_group_by_case = _maybe_load_spacing_groups(summary)

    rows_raw = _read_json(metrics_p)
    if not isinstance(rows_raw, list):
        raise ValueError(f"metrics.json must be a list: {metrics_p}")

    rows = [_extract_case_row(r, thr) for r in rows_raw]
    if spacing_group_by_case:
        for r in rows:
            cid = r.get("case_id")
            if cid and cid in spacing_group_by_case:
                r["slice_spacing_group"] = spacing_group_by_case[cid]
    # attach slice spacing group if available from summary bins (only aggregate exists); we rely on metrics.json row keys when present.
    return thr, {"split": split, "summary": summary}, rows


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def _p(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = int(round((len(xs) - 1) * q))
    return float(xs[i])


def _group_key(row: dict[str, Any]) -> str:
    # prefer explicit le_3mm/gt_3mm grouping if present
    for k in ("slice_spacing_group", "slice_spacing_bin", "slice_spacing"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return "unknown"


def _size_bucket(gt_vox: float | None, bins: list[int]) -> str:
    if gt_vox is None:
        return "unknown"
    if gt_vox <= bins[0]:
        return f"le_{bins[0]}"
    if gt_vox <= bins[1]:
        return f"le_{bins[1]}"
    return f"gt_{bins[1]}"


@dataclass
class DiffRow:
    case_id: str
    dice_a: float
    dice_b: float
    delta: float
    gt_vox: float | None
    group: str
    size: str


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two eval runs and summarize failure modes.")
    ap.add_argument("--run-a", type=str, required=True, help="baseline eval dir containing summary.json + metrics.json")
    ap.add_argument("--run-b", type=str, required=True, help="candidate eval dir containing summary.json + metrics.json")
    ap.add_argument("--name-a", type=str, default="A")
    ap.add_argument("--name-b", type=str, default="B")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    run_a = Path(args.run_a).expanduser().resolve()
    run_b = Path(args.run_b).expanduser().resolve()

    thr_a, meta_a, rows_a = _load_run(run_a)
    thr_b, meta_b, rows_b = _load_run(run_b)

    # Use each run's own best threshold for its dice extraction. This matches how the summaries were reported.
    a_by = {r["case_id"]: r for r in rows_a if r.get("case_id")}
    b_by = {r["case_id"]: r for r in rows_b if r.get("case_id")}
    common = sorted(set(a_by).intersection(set(b_by)))

    bins = list(meta_a["summary"].get("gt_size_bins") or [250, 1000])
    if len(bins) != 2:
        bins = [250, 1000]

    diffs: list[DiffRow] = []
    for cid in common:
        ra, rb = a_by[cid], b_by[cid]
        da = float(ra.get("dice") or 0.0)
        db = float(rb.get("dice") or 0.0)
        dv = db - da
        gt = _safe_float(ra.get("gt_vox"))
        group = _group_key(ra) if _group_key(ra) != "unknown" else _group_key(rb)
        size = _size_bucket(gt, bins)
        diffs.append(DiffRow(case_id=cid, dice_a=da, dice_b=db, delta=dv, gt_vox=gt, group=group, size=size))

    diffs_sorted = sorted(diffs, key=lambda r: r.delta)
    worst = diffs_sorted[: max(0, int(args.top_k))]
    best = list(reversed(diffs_sorted[-max(0, int(args.top_k)) :]))

    def summarize(sub: list[DiffRow]) -> dict[str, Any]:
        deltas = [r.delta for r in sub]
        return {
            "n": len(sub),
            "mean_delta": _mean(deltas) if deltas else 0.0,
            "p10": _p(deltas, 0.10),
            "p50": _p(deltas, 0.50),
            "p90": _p(deltas, 0.90),
        }

    by_group: dict[str, list[DiffRow]] = {}
    by_size: dict[str, list[DiffRow]] = {}
    for r in diffs:
        by_group.setdefault(r.group, []).append(r)
        by_size.setdefault(r.size, []).append(r)

    report: dict[str, Any] = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "split_a": meta_a["split"],
        "split_b": meta_b["split"],
        "best_thr_a": thr_a,
        "best_thr_b": thr_b,
        "n_common": len(common),
        "overall": summarize(diffs),
        "by_group": {k: summarize(v) for k, v in sorted(by_group.items(), key=lambda kv: kv[0])},
        "by_size": {k: summarize(v) for k, v in sorted(by_size.items(), key=lambda kv: kv[0])},
        "worst_cases": [
            {"case_id": r.case_id, "delta": r.delta, "dice_a": r.dice_a, "dice_b": r.dice_b, "group": r.group, "size": r.size, "gt_vox": r.gt_vox}
            for r in worst
        ],
        "best_cases": [
            {"case_id": r.case_id, "delta": r.delta, "dice_a": r.dice_a, "dice_b": r.dice_b, "group": r.group, "size": r.size, "gt_vox": r.gt_vox}
            for r in best
        ],
    }

    out_path = Path(args.out).expanduser().resolve() if args.out else None
    text = json.dumps(report, indent=2)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
