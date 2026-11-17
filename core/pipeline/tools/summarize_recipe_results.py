from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _find_latest_results_json(eval_out_root: Path) -> Path | None:
    logs_root = eval_out_root / "recipe_queue_logs"
    if not logs_root.exists():
        return None
    cands = list(logs_root.glob("queue_*/results.json"))
    if not cands:
        # fallback: sometimes user may only have jsonl
        cands = list(logs_root.glob("queue_*/results.jsonl"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


_EVAL_DIR_RE = re.compile(r"^test_eval_(?P<tta>[^_]+)_(?P<exp>.+)_(?P<stamp>\d{8}_\d{6})$")


def _load_records_from_eval_out_root(eval_out_root: Path) -> list[dict]:
    # Support using this tool even when run_recipe_queue.py wasn't used.
    # Collect per-eval directories produced by tools/wait_train_and_eval.py:
    #   <eval_out_root>/test_eval_<tta>_<exp>_<stamp>/best_thresholds.json
    if not eval_out_root.exists():
        return []

    def best_row(summary: dict, bucket: str) -> dict:
        per = summary.get("per_threshold") or []
        if not per:
            raise ValueError("summary.json has empty per_threshold")

        if bucket == "all":
            return max(per, key=lambda r: float(r.get("mean_dice") or -1.0))

        def bval(r: dict) -> float:
            by = (r.get("by_slice_spacing") or {}).get(bucket) or {}
            return float(by.get("mean_dice") or -1.0)

        return max(per, key=bval)

    def row_to_dict(row: dict) -> dict:
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

    recs: list[dict] = []
    for d in sorted(eval_out_root.iterdir()):
        if not d.is_dir():
            continue
        m = _EVAL_DIR_RE.match(d.name)
        if not m:
            continue
        best_p = d / "best_thresholds.json"
        if best_p.exists() and best_p.stat().st_size > 0:
            obj = json.loads(best_p.read_text(encoding="utf-8"))
        else:
            # Older runs may not have best_thresholds.json; compute from summary.json.
            sp = d / "summary.json"
            if not sp.exists() or sp.stat().st_size == 0:
                continue
            summary = json.loads(sp.read_text(encoding="utf-8"))
            r_all = best_row(summary, "all")
            r_le = best_row(summary, "le_3mm")
            obj = {"best_all": row_to_dict(r_all), "best_le_3mm": row_to_dict(r_le)}
        recs.append(
            {
                "timestamp": m.group("stamp"),
                "exp_name": m.group("exp"),
                "split": "unknown",
                "config": "",
                "best_thresholds_path": str(best_p if best_p.exists() else (d / "summary.json")),
                "best_thresholds": obj,
                "tta": m.group("tta"),
            }
        )

    return recs


def _get(obj: dict, path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _load_records(p: Path) -> list[dict]:
    if p.name.endswith(".jsonl"):
        recs: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
        return recs

    obj = json.loads(p.read_text(encoding="utf-8"))
    runs = obj.get("runs")
    if isinstance(runs, list):
        return runs
    raise ValueError(f"Unsupported JSON format: {p}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarize recipe queue results (best_all threshold per run) into a compact table."
    )
    ap.add_argument(
        "--input",
        default="",
        help="Path to results.json or results.jsonl. If omitted, auto-detect from --eval-out-root.",
    )
    ap.add_argument(
        "--eval-out-root",
        default="results/diag/_recipe_20251227",
        help="Used for auto-detect when --input is omitted.",
    )
    ap.add_argument("--format", choices=["tsv", "md", "json"], default="tsv")
    ap.add_argument("--sort", choices=["best_all", "le_3mm"], default="best_all")
    ap.add_argument("--show-best-le", action="store_true", help="Also show best_le_3mm fields")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    in_path: Path | None
    if args.input:
        in_path = Path(args.input)
        if not in_path.is_absolute():
            in_path = (repo_root / in_path).resolve()
    else:
        eval_out_root = Path(args.eval_out_root)
        if not eval_out_root.is_absolute():
            eval_out_root = (repo_root / eval_out_root).resolve()
        in_path = _find_latest_results_json(eval_out_root)

    if in_path is None or not in_path.exists():
        # Fallback: scan eval_out_root directly.
        eval_out_root = Path(args.eval_out_root)
        if not eval_out_root.is_absolute():
            eval_out_root = (repo_root / eval_out_root).resolve()
        recs = _load_records_from_eval_out_root(eval_out_root)
        if not recs:
            print(
                "No results found yet. Run tools/run_recipe_queue.py first, or pass --input to an existing results.json(l).",
                flush=True,
            )
            return 0
        in_path = eval_out_root
    else:
        recs = _load_records(in_path)

    rows: list[dict[str, Any]] = []
    for r in recs:
        bt = r.get("best_thresholds") or {}
        rows.append(
            {
                "exp_name": r.get("exp_name"),
                "split": r.get("split"),
                "config": r.get("config"),
                "best_all_thr": _as_float(_get(bt, "best_all.threshold")),
                "best_all_dice": _as_float(_get(bt, "best_all.mean_dice")),
                "best_all_le_dice": _as_float(_get(bt, "best_all.by_slice_spacing.le_3mm.mean_dice")),
                "best_all_le_det": _as_float(
                    _get(bt, "best_all.by_slice_spacing.le_3mm.detection_rate_case")
                ),
                "best_all_gt_dice": _as_float(_get(bt, "best_all.by_slice_spacing.gt_3mm.mean_dice")),
                "best_all_gt_det": _as_float(
                    _get(bt, "best_all.by_slice_spacing.gt_3mm.detection_rate_case")
                ),
                "best_le_thr": _as_float(_get(bt, "best_le_3mm.threshold")),
                "best_le_dice": _as_float(_get(bt, "best_le_3mm.mean_dice")),
                "best_le_le_dice": _as_float(_get(bt, "best_le_3mm.by_slice_spacing.le_3mm.mean_dice")),
                "best_le_le_det": _as_float(
                    _get(bt, "best_le_3mm.by_slice_spacing.le_3mm.detection_rate_case")
                ),
            }
        )

    if args.sort == "best_all":
        rows.sort(key=lambda x: float(x.get("best_all_dice") or -1.0), reverse=True)
    else:
        rows.sort(key=lambda x: float(x.get("best_all_le_dice") or -1.0), reverse=True)

    if args.format == "json":
        print(json.dumps({"input": str(in_path), "rows": rows}, ensure_ascii=False, indent=2))
        return 0

    if args.show_best_le:
        header = [
            "exp_name",
            "split",
            "best_all_thr",
            "best_all_dice",
            "best_all_le_dice",
            "best_all_le_det",
            "best_all_gt_dice",
            "best_all_gt_det",
            "best_le_thr",
            "best_le_dice",
            "best_le_le_dice",
            "best_le_le_det",
        ]
    else:
        header = [
            "exp_name",
            "split",
            "best_all_thr",
            "best_all_dice",
            "best_all_le_dice",
            "best_all_le_det",
            "best_all_gt_dice",
            "best_all_gt_det",
        ]

    if args.format == "md":
        print(f"# recipe results: {in_path}")
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            vals = [row.get(k) for k in header]
            vals2 = ["" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in vals]
            print("| " + " | ".join(vals2) + " |")
        return 0

    # tsv
    print("\t".join(header))
    for row in rows:
        vals = [row.get(k) for k in header]
        out: list[str] = []
        for v in vals:
            if v is None:
                out.append("")
            elif isinstance(v, float):
                out.append(f"{v:.6f}")
            else:
                out.append(str(v))
        print("\t".join(out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
