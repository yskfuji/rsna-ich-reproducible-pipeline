from __future__ import annotations

import argparse
import json
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(str(path))
    last: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        last = json.loads(line)
    if last is None:
        raise ValueError(f"Empty jsonl: {path}")
    return last


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class FoldRow:
    fold: int
    out_dir: str
    metrics: dict[str, float]


def _find_fold_dirs(cv_root: Path) -> list[Path]:
    if not cv_root.exists():
        return []

    fold_dirs = [p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold")]

    def key(p: Path) -> tuple[int, str]:
        suf = p.name.replace("fold", "")
        try:
            return (int(suf), p.name)
        except Exception:
            return (10**9, p.name)

    fold_dirs.sort(key=key)
    return fold_dirs


def _fmt_float(x: float | None) -> str:
    if x is None:
        return ""
    # Keep compact but stable.
    return f"{x:.8f}".rstrip("0").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize RSNA CV runs (fold*/log.jsonl last line) into mean/std/min/max for chosen metrics."
        )
    )
    ap.add_argument(
        "--cv-root",
        required=True,
        help='CV output root directory containing fold subdirectories (e.g. results/<YOUR_CV_DIR>).',
    )
    ap.add_argument(
        "--metrics",
        default="val_logloss_weighted,val_auc_mean,val_loss_plain,val_loss",
        help="Comma-separated metric keys to extract from each fold's log.jsonl last line.",
    )
    ap.add_argument("--format", choices=["md", "tsv", "json"], default="md")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested metric is missing in any fold.",
    )

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cv_root = Path(args.cv_root)
    if not cv_root.is_absolute():
        cv_root = (repo_root / cv_root).resolve()

    metric_keys = [k.strip() for k in str(args.metrics).split(",") if k.strip()]
    if not metric_keys:
        print("No metrics specified.", flush=True)
        return 2

    fold_dirs = _find_fold_dirs(cv_root)
    if not fold_dirs:
        print(f"No fold directories found under: {cv_root}", flush=True)
        return 2

    rows: list[FoldRow] = []
    for d in fold_dirs:
        meta_p = d / "meta.json"
        log_p = d / "log.jsonl"
        meta = _read_json(meta_p)
        log_last = _read_last_jsonl(log_p)

        fold_index = meta.get("cv_fold_index")
        try:
            fold_i = int(fold_index) if fold_index is not None else int(d.name.replace("fold", ""))
        except Exception:
            fold_i = int(d.name.replace("fold", "")) if d.name.replace("fold", "").isdigit() else 0

        metrics: dict[str, float] = {}
        missing: list[str] = []
        for k in metric_keys:
            v = _as_float(log_last.get(k))
            if v is None:
                missing.append(k)
                continue
            metrics[k] = float(v)

        if missing and args.strict:
            raise KeyError(f"Missing metrics in {d}: {missing}")

        rows.append(FoldRow(fold=fold_i, out_dir=str(d), metrics=metrics))

    rows.sort(key=lambda r: r.fold)

    # Aggregate.
    summary: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        vals = [r.metrics.get(k) for r in rows if k in r.metrics]
        vals_f = [float(v) for v in vals if v is not None]
        if not vals_f:
            continue
        mean = sum(vals_f) / len(vals_f)
        std = stats.stdev(vals_f) if len(vals_f) >= 2 else 0.0
        summary[k] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(min(vals_f)),
            "max": float(max(vals_f)),
            "n_folds": float(len(vals_f)),
        }

    if args.format == "json":
        obj = {
            "cv_root": str(cv_root),
            "folds": [
                {
                    "fold": r.fold,
                    "out_dir": r.out_dir,
                    "metrics": r.metrics,
                }
                for r in rows
            ],
            "summary": summary,
            "metrics": metric_keys,
        }
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return 0

    if args.format == "tsv":
        header = ["fold"] + metric_keys
        print("\t".join(header))
        for r in rows:
            parts = [str(r.fold)]
            for k in metric_keys:
                parts.append(_fmt_float(r.metrics.get(k)))
            print("\t".join(parts))
        print("\n# summary")
        print("metric\tmean\tstd\tmin\tmax\tn_folds")
        for k in metric_keys:
            s = summary.get(k)
            if not s:
                continue
            print(
                "\t".join(
                    [
                        k,
                        _fmt_float(s.get("mean")),
                        _fmt_float(s.get("std")),
                        _fmt_float(s.get("min")),
                        _fmt_float(s.get("max")),
                        str(int(s.get("n_folds") or 0)),
                    ]
                )
            )
        return 0

    # md
    print(f"cv_root: {cv_root}")
    print("")
    print("| fold | " + " | ".join(metric_keys) + " |")
    print("|---:|" + "|".join(["---:"] * len(metric_keys)) + "|")
    for r in rows:
        cells = [str(r.fold)] + [_fmt_float(r.metrics.get(k)) for k in metric_keys]
        print("| " + " | ".join(cells) + " |")

    print("")
    print("| metric | mean | std | min | max | n_folds |")
    print("|:--|--:|--:|--:|--:|---:|")
    for k in metric_keys:
        s = summary.get(k)
        if not s:
            continue
        print(
            "| "
            + " | ".join(
                [
                    k,
                    _fmt_float(s.get("mean")),
                    _fmt_float(s.get("std")),
                    _fmt_float(s.get("min")),
                    _fmt_float(s.get("max")),
                    str(int(s.get("n_folds") or 0)),
                ]
            )
            + " |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
