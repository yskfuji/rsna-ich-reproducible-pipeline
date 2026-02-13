#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make `src` importable regardless of current working directory.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.datasets.rsna_ich_dataset import (  # noqa: E402
    RSNA_CLASSES,
    RsnaSliceRecord,
    iter_rsna_stage2_records_from_csv,
    preprocessed_db_existing_keys,
    preprocessed_db_has_meta,
    read_rsna_preprocessed_meta,
)


def _read_study_and_series_uid(dcm_path: Path) -> tuple[str | None, str | None]:
    import warnings

    from pydicom import dcmread  # type: ignore

    warnings.filterwarnings("ignore", message=r"Invalid value for VR UI:.*", module=r"pydicom\..*")
    try:
        ds = dcmread(  # type: ignore[reportUnknownMemberType]
            str(dcm_path),
            stop_before_pixels=True,
            specific_tags=["StudyInstanceUID", "SeriesInstanceUID"],
            force=True,
        )
    except Exception:
        return None, None

    study_uid = str(getattr(ds, "StudyInstanceUID", "") or "").strip() or None
    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip() or None
    return study_uid, series_uid


def _group_split_indices(
    *,
    records: list[RsnaSliceRecord],
    group_ids: list[str],
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    if len(records) != len(group_ids):
        raise ValueError("records and group_ids length mismatch")

    groups: dict[str, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(str(g), []).append(i)

    rng = np.random.default_rng(int(seed))
    all_groups = list(groups.keys())

    any_idx = int(RSNA_CLASSES.index("any"))

    def _is_pos_group(idxs: list[int]) -> bool:
        return any(float(records[i].y[any_idx]) > 0.5 for i in idxs)

    pos_groups: list[str] = []
    neg_groups: list[str] = []
    for g in all_groups:
        (pos_groups if _is_pos_group(groups[g]) else neg_groups).append(g)

    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    n_groups = len(all_groups)
    n_val_groups = int(max(1, round(float(val_frac) * n_groups)))
    n_val_groups = int(min(n_val_groups, n_groups - 1)) if n_groups >= 2 else 1

    pos_target = int(round(n_val_groups * (len(pos_groups) / max(1, n_groups))))
    pos_target = int(min(max(0, pos_target), len(pos_groups)))
    neg_target = int(n_val_groups - pos_target)
    neg_target = int(min(max(0, neg_target), len(neg_groups)))
    while pos_target + neg_target < n_val_groups:
        if len(pos_groups) - pos_target > len(neg_groups) - neg_target:
            if pos_target < len(pos_groups):
                pos_target += 1
                continue
        if neg_target < len(neg_groups):
            neg_target += 1
            continue
        break

    val_group_set = set(pos_groups[:pos_target] + neg_groups[:neg_target])
    if not val_group_set and all_groups:
        val_group_set.add(all_groups[0])

    tr_idx: list[int] = []
    va_idx: list[int] = []
    for g, idxs in groups.items():
        (va_idx if g in val_group_set else tr_idx).extend(idxs)

    stats: dict[str, Any] = {
        "split_groups_total": n_groups,
        "split_groups_val": len(val_group_set),
        "split_groups_train": len(all_groups) - len(val_group_set),
        "split_groups_pos_total": len(pos_groups),
        "split_groups_neg_total": len(neg_groups),
        "split_groups_pos_val": sum(1 for g in val_group_set if g in set(pos_groups)),
        "split_groups_neg_val": sum(1 for g in val_group_set if g in set(neg_groups)),
    }

    return tr_idx, va_idx, stats


def _resolve_path(p: str | None) -> Path | None:
    if p is None:
        return None
    v = str(p).strip()
    if not v:
        return None
    return Path(v).expanduser().resolve()


def _build_group_ids(
    *,
    records: list[RsnaSliceRecord],
    split_by: str,
    pre_db: Path | None,
) -> list[str]:
    split_by_s = str(split_by).strip().lower()
    if split_by_s not in {"study", "series"}:
        raise ValueError("split_by must be 'study' or 'series' for group split audit")

    group_ids: list[str] = []
    if pre_db is None:
        cache: dict[str, tuple[str | None, str | None]] = {}
        for r in records:
            key = str(getattr(r, "dcm_path"))
            if key not in cache:
                cache[key] = _read_study_and_series_uid(Path(key))
            study_uid, series_uid = cache[key]
            if split_by_s == "study":
                group_ids.append(study_uid or series_uid or str(getattr(r, "image_id")))
            else:
                group_ids.append(series_uid or study_uid or str(getattr(r, "image_id")))
        return group_ids

    if not preprocessed_db_has_meta(pre_db):
        raise SystemExit(
            "preprocessed DB has no meta table. Rebuild/backfill via tools/precompute_rsna_preprocessed_sqlite.py"
        )

    cache2: dict[str, tuple[str | None, str | None]] = {}
    for r in records:
        img = str(getattr(r, "image_id"))
        if img not in cache2:
            study_uid, series_uid, _inst, _series_idx = read_rsna_preprocessed_meta(image_id=img, db_path=pre_db)
            cache2[img] = (study_uid, series_uid)
        study_uid, series_uid = cache2[img]
        if split_by_s == "study":
            group_ids.append(study_uid or series_uid or img)
        else:
            group_ids.append(series_uid or study_uid or img)

    return group_ids


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Audit RSNA validation split leakage by verifying group (study/series) disjointness. "
            "Uses the exact same split logic as src/training/train_rsna_cnn2d_classifier.py."
        )
    )
    p.add_argument(
        "--rsna-root",
        required=True,
        type=str,
        help="RSNA root containing stage_2_train.csv and stage_2_train/ (or <preprocessed_root>/rsna_meta)",
    )
    p.add_argument(
        "--preprocessed-root",
        default=None,
        type=str,
        help="If set, uses <preprocessed_root>/train.sqlite meta (faster; no DICOM needed)",
    )
    p.add_argument("--limit-images", default=8000, type=int, help="Number of unique image_ids to sample")
    p.add_argument("--val-frac", default=0.05, type=float, help="Validation fraction")
    p.add_argument("--split-by", default="study", type=str, help="Group split unit: study | series")
    p.add_argument("--seed", default=0, type=int, help="Random seed")
    p.add_argument("--out-json", default=None, type=str, help="Optional output path for JSON report")

    ns = p.parse_args(argv)

    rsna_root = _resolve_path(ns.rsna_root)
    if rsna_root is None:
        raise SystemExit("--rsna-root is required")

    pre_root = _resolve_path(ns.preprocessed_root)
    pre_db: Path | None = None
    if pre_root is not None:
        pre_db = pre_root / "train.sqlite"
        if not pre_db.exists():
            raise SystemExit(f"Missing preprocessed DB: {pre_db}")

    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    records: list[RsnaSliceRecord] = iter_rsna_stage2_records_from_csv(
        csv_path=csv_path,
        dicom_dir=dcm_dir,
        limit_images=int(ns.limit_images) if int(ns.limit_images) > 0 else None,
        seed=int(ns.seed),
    )

    if pre_db is not None:
        exist = preprocessed_db_existing_keys([str(r.image_id) for r in records], db_path=pre_db)
        before = len(records)
        records = [r for r in records if str(r.image_id) in exist]
        dropped = before - len(records)
        if dropped:
            print(
                f"[preprocessed] filtered missing keys: kept={len(records)} dropped={dropped}",
                file=sys.stderr,
                flush=True,
            )

    group_ids = _build_group_ids(records=records, split_by=str(ns.split_by), pre_db=pre_db)

    tr_idx, va_idx, stats = _group_split_indices(
        records=records,
        group_ids=group_ids,
        val_frac=float(ns.val_frac),
        seed=int(ns.seed),
    )

    tr_groups = {group_ids[i] for i in tr_idx}
    va_groups = {group_ids[i] for i in va_idx}
    inter_groups = tr_groups & va_groups

    tr_img = {str(records[i].image_id) for i in tr_idx}
    va_img = {str(records[i].image_id) for i in va_idx}
    inter_img = tr_img & va_img

    report: dict[str, Any] = {
        "rsna_root": str(rsna_root),
        "preprocessed_root": str(pre_root) if pre_root is not None else None,
        "limit_images": int(ns.limit_images),
        "val_frac": float(ns.val_frac),
        "split_by": str(ns.split_by).strip().lower(),
        "seed": int(ns.seed),
        "n_records": int(len(records)),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "split_stats": stats,
        "n_train_groups": int(len(tr_groups)),
        "n_val_groups": int(len(va_groups)),
        "n_group_intersection": int(len(inter_groups)),
        "n_imageid_intersection": int(len(inter_img)),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if ns.out_json:
        out = Path(str(ns.out_json)).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[saved] {out}")

    # Non-zero exit if leakage detected.
    if report["n_group_intersection"] != 0 or report["n_imageid_intersection"] != 0:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
