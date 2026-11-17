#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _resolve_path(p: str | None) -> Path | None:
    if p is None:
        return None
    return Path(p).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Audit leakage symptoms for split_by=slice by counting train/val overlaps of Study/Series. "
            "This reproduces the same record sampling + slice split behavior as src/training/train_rsna_cnn2d_classifier.py."
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
        required=True,
        type=str,
        help="Preprocessed root containing train.sqlite with meta table (for Study/Series UIDs)",
    )
    p.add_argument("--limit-images", default=8000, type=int, help="Number of unique image_ids to sample")
    p.add_argument("--val-frac", default=0.05, type=float, help="Validation fraction")
    p.add_argument("--seed", default=0, type=int, help="Random seed")
    p.add_argument(
        "--out-json",
        default=None,
        type=str,
        help="If set, also write the report JSON to this path (stdout is always JSON)",
    )

    args = p.parse_args(argv)

    # Make `src` importable regardless of current working directory.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.datasets.rsna_ich_dataset import (  # noqa: E402
        iter_rsna_stage2_records_from_csv,
        preprocessed_db_existing_keys,
        preprocessed_db_has_meta,
        read_rsna_preprocessed_meta,
    )

    rsna_root = _resolve_path(args.rsna_root)
    if rsna_root is None:
        raise ValueError("--rsna-root is required")
    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing stage_2_train.csv: {csv_path}")

    pre_root = _resolve_path(args.preprocessed_root)
    if pre_root is None:
        raise ValueError("--preprocessed-root is required")
    pre_db = pre_root / "train.sqlite"
    if not pre_db.exists():
        raise FileNotFoundError(f"Missing preprocessed DB: {pre_db}")
    if not preprocessed_db_has_meta(pre_db):
        raise SystemExit(
            "preprocessed DB has no meta table. Rebuild/backfill via tools/precompute_rsna_preprocessed_sqlite.py"
        )

    val_frac = float(args.val_frac)
    if val_frac < 0.0 or val_frac >= 1.0:
        raise ValueError("val_frac must be in [0, 1)")

    records = list(
        iter_rsna_stage2_records_from_csv(
            csv_path=csv_path,
            dicom_dir=dcm_dir,
            limit_images=int(args.limit_images) if int(args.limit_images) > 0 else None,
            seed=int(args.seed),
        )
    )

    exist = preprocessed_db_existing_keys([r.image_id for r in records], db_path=pre_db)
    before = len(records)
    records = [r for r in records if r.image_id in exist]
    dropped = before - len(records)
    if dropped:
        print(f"[preprocessed] filtered missing keys: kept={len(records)} dropped={dropped}", file=sys.stderr, flush=True)
    if not records:
        raise FileNotFoundError(f"No matching keys found in preprocessed DB: {pre_db}")

    n = int(len(records))
    n_val = int(max(1, round(val_frac * n)))
    n_tr = int(max(1, n - n_val))
    train_records = records[:n_tr]
    val_records = records[n_tr:]

    cache: dict[str, tuple[str | None, str | None]] = {}

    def _get_uids(image_id: str) -> tuple[str | None, str | None]:
        if image_id not in cache:
            study_uid, series_uid, _inst, _series_idx = read_rsna_preprocessed_meta(image_id=image_id, db_path=pre_db)
            cache[image_id] = (study_uid, series_uid)
        return cache[image_id]

    tr_img = {str(r.image_id) for r in train_records}
    va_img = {str(r.image_id) for r in val_records}
    img_inter = tr_img & va_img

    tr_study: set[str] = set()
    va_study: set[str] = set()
    tr_series: set[str] = set()
    va_series: set[str] = set()

    for img in tr_img:
        s, se = _get_uids(img)
        if s:
            tr_study.add(str(s))
        if se:
            tr_series.add(str(se))
    for img in va_img:
        s, se = _get_uids(img)
        if s:
            va_study.add(str(s))
        if se:
            va_series.add(str(se))

    study_inter = tr_study & va_study
    series_inter = tr_series & va_series

    report: dict[str, Any] = {
        "split_by": "slice",
        "val_frac_requested": float(val_frac),
        "val_frac_effective": float(len(val_records) / max(1, len(records))),
        "seed": int(args.seed),
        "limit_images": int(args.limit_images),
        "n_records": int(len(records)),
        "n_train": int(len(train_records)),
        "n_val": int(len(val_records)),
        "n_imageids_train": int(len(tr_img)),
        "n_imageids_val": int(len(va_img)),
        "n_imageid_intersection": int(len(img_inter)),
        "n_train_studies": int(len(tr_study)),
        "n_val_studies": int(len(va_study)),
        "n_study_intersection": int(len(study_inter)),
        "frac_val_studies_in_train": float(len(study_inter) / max(1, len(va_study))),
        "n_train_series": int(len(tr_series)),
        "n_val_series": int(len(va_series)),
        "n_series_intersection": int(len(series_inter)),
        "frac_val_series_in_train": float(len(series_inter) / max(1, len(va_series))),
    }

    out_json = _resolve_path(args.out_json)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[saved] {out_json}", file=sys.stderr, flush=True)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
