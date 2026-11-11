"""Export the public metric summary (reports/rsna_cv_summary.json).

Reads the private run directories (fold*/log.jsonl, fold*/meta.json and the
MC-dropout sweep_summary.json) and writes only per-fold / per-seed scalar
metrics and run settings. No image IDs, file paths or tensors are exported.

Example:
    python tools/export_metric_summary.py \
        --cv-root results/rsna_convnext25d_ft_gkfold5_e1_lim8k_20260210_150447 \
        --sweep-root results/mc_dropout_retrain/sweep_seed0to9_dedup_e1_lim8k_val05 \
        --init-meta results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/meta.json \
        --out reports/rsna_cv_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CV_METRICS = ("val_logloss_weighted", "val_auc_mean")
UNCERTAINTY_METRICS = (
    "auroc_uncertainty_detect_error_any",
    "ece_any",
    "brier_any",
    "aurc_weighted_logloss",
    "accuracy_any_improve_pp",
)
RUN_SETTINGS = (
    "arch",
    "epochs",
    "limit_images",
    "split_by",
    "stack_slices",
    "pretrained",
    "enforce_any_max",
    "torch_version",
)


def _last_jsonl(path: Path) -> dict[str, Any]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty jsonl: {path}")
    return json.loads(lines[-1])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _init_name(meta: dict[str, Any]) -> str | None:
    init_from = meta.get("init_from")
    if not init_from:
        return None
    p = Path(str(init_from))
    return f"{p.parent.name}/{p.name}"


def export_cv(cv_root: Path) -> dict[str, Any]:
    fold_dirs = sorted(
        (p for p in cv_root.iterdir() if p.is_dir() and re.fullmatch(r"fold\d+", p.name)),
        key=lambda p: int(p.name.removeprefix("fold")),
    )
    folds: list[dict[str, Any]] = []
    settings: dict[str, Any] | None = None
    for fd in fold_dirs:
        last = _last_jsonl(fd / "log.jsonl")
        meta = _read_json(fd / "meta.json")
        split = meta.get("split_stats") or {}
        folds.append(
            {
                "fold": int(fd.name.removeprefix("fold")),
                "n_train": int(meta["n_train"]),
                "n_val": int(meta["n_val"]),
                "val_groups": int(split.get("split_groups_val", 0)),
                **{k: float(last[k]) for k in CV_METRICS},
            }
        )
        if settings is None:
            settings = {k: meta.get(k) for k in RUN_SETTINGS}
            settings["init_from"] = _init_name(meta)
            settings["split_mode"] = split.get("split_mode")
            settings["cv_folds"] = split.get("cv_folds")
    return {"run": cv_root.name, "settings": settings, "folds": folds}


def export_uncertainty(sweep_root: Path) -> dict[str, Any]:
    summary = _read_json(sweep_root / "sweep_summary.json")
    seeds: list[dict[str, Any]] = []
    settings: dict[str, Any] | None = None
    for row in summary["per_seed"]:
        seed = int(row["seed"])
        meta = _read_json(sweep_root / f"seed{seed}" / "meta.json")
        arm = row["retrained_with_dropout"]
        base = row["baseline"]
        seeds.append(
            {
                "seed": seed,
                "n_val": int(arm["n_val"]),
                "temperature": float(arm["temperature"]),
                "retrained_with_dropout": {k: float(arm[k]) for k in UNCERTAINTY_METRICS},
                "baseline": {k: float(base[k]) for k in UNCERTAINTY_METRICS},
            }
        )
        if settings is None:
            settings = {k: meta.get(k) for k in RUN_SETTINGS}
            settings["init_from"] = _init_name(meta)
            settings["dedup_before_split"] = meta.get("dedup_before_split")
            settings["mc_samples"] = int(arm["mc_samples"])
            settings["dropout_stage_p"] = float(arm["dropout_stage_p"])
            settings["dropout_head_p"] = float(arm["dropout_head_p"])
            settings["coverage_point"] = float(arm["coverage_point"])
            # Not recorded in the run outputs. The sweep never passes --ece-bins, so
            # eval_rsna_uncertainty.py uses its default of 15, and its --fit-temperature
            # fits on the same split it evaluates.
            settings["ece_bins"] = 15
            settings["temperature_fit_set"] = "same holdout as evaluation"
            settings["settings_from_code"] = ["ece_bins", "temperature_fit_set"]
    if settings is not None:
        settings["temperature_fitted"] = any(abs(s["temperature"] - 1.0) > 1e-6 for s in seeds)
    return {"run": sweep_root.name, "settings": settings, "seeds": seeds}


def export_init(init_meta: Path) -> dict[str, Any]:
    meta = _read_json(init_meta)
    return {
        "run": init_meta.parent.name,
        "n_train": int(meta["n_train"]),
        "n_val": int(meta["n_val"]),
        "val_frac": float(meta["val_frac"]),
        "split_by": meta.get("split_by"),
        "epochs": int(meta["epochs"]),
        "limit_images": int(meta.get("limit_images") or 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv-root", type=Path, required=True)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--init-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args()

    payload = {
        "schema_version": 1,
        "description": "Per-fold / per-seed scalar metrics behind the headline numbers in the READMEs.",
        "aggregation": {"center": "mean", "spread": "sample standard deviation (ddof=1)"},
        "cross_validation": export_cv(ns.cv_root),
        "uncertainty": export_uncertainty(ns.sweep_root),
        "init_checkpoint": export_init(ns.init_meta),
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
