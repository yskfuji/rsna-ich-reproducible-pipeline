#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.datasets.rsna_ich_dataset import (
    RSNA_CLASSES,
    iter_rsna_stage2_records_from_csv,
    preprocessed_db_existing_keys,
    read_rsna_preprocessed_meta,
)
from src.training.train_rsna_cnn2d_classifier import _group_split_records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Adversarial leakage audit for RSNA split_by=study. "
            "Tests hypothesis: train/val leakage exists."
        )
    )
    p.add_argument("--rsna-root", required=True, type=str)
    p.add_argument("--preprocessed-root", required=True, type=str)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)
    p.add_argument("--limit-images", default=8000, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)
    p.add_argument("--out-json", default="/tmp/rsna_leakage_hypothesis_audit.json", type=str)
    ns = p.parse_args(argv)

    seeds = [int(x.strip()) for x in str(ns.seeds).split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds is empty")

    rsna_root = Path(str(ns.rsna_root)).expanduser().resolve()
    pre_root = Path(str(ns.preprocessed_root)).expanduser().resolve()
    pre_db = pre_root / "train.sqlite"

    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    any_idx = int(RSNA_CLASSES.index("any"))

    def _pos_group_selector(records: list[Any]) -> bool:
        return any(float(r.y[any_idx]) > 0.5 for r in records)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        records = iter_rsna_stage2_records_from_csv(
            csv_path=csv_path,
            dicom_dir=dcm_dir,
            limit_images=int(ns.limit_images) if int(ns.limit_images) > 0 else None,
            seed=int(seed),
        )

        exist = preprocessed_db_existing_keys([str(r.image_id) for r in records], db_path=pre_db)
        records = [r for r in records if str(r.image_id) in exist]

        meta: dict[str, tuple[str | None, str | None]] = {}
        group_ids: list[str] = []

        n_missing_study = 0
        n_missing_series = 0
        n_fallback_to_series = 0
        n_fallback_to_image = 0

        for r in records:
            img = str(r.image_id)
            if img not in meta:
                study_uid, series_uid, _inst, _series_idx = read_rsna_preprocessed_meta(image_id=img, db_path=pre_db)
                meta[img] = (study_uid, series_uid)
            study_uid, series_uid = meta[img]

            if not study_uid:
                n_missing_study += 1
            if not series_uid:
                n_missing_series += 1

            if study_uid:
                gid = study_uid
            elif series_uid:
                gid = series_uid
                n_fallback_to_series += 1
            else:
                gid = img
                n_fallback_to_image += 1
            group_ids.append(gid)

        train_records, val_records, split_stats = _group_split_records(
            records=records,
            group_ids=group_ids,
            val_frac=float(ns.val_frac),
            seed=int(seed),
            pos_group_selector=_pos_group_selector,
        )

        tr_img = {str(r.image_id) for r in train_records}
        va_img = {str(r.image_id) for r in val_records}
        img_inter = tr_img & va_img

        def _uid_sets(rs: list[Any]) -> tuple[set[str], set[str]]:
            s_study: set[str] = set()
            s_series: set[str] = set()
            for r in rs:
                st, se = meta[str(r.image_id)]
                if st:
                    s_study.add(st)
                if se:
                    s_series.add(se)
            return s_study, s_series

        tr_study, tr_series = _uid_sets(train_records)
        va_study, va_series = _uid_sets(val_records)

        rows.append(
            {
                "seed": int(seed),
                "n_records": int(len(records)),
                "n_train": int(len(train_records)),
                "n_val": int(len(val_records)),
                "n_image_intersection": int(len(img_inter)),
                "n_study_intersection": int(len(tr_study & va_study)),
                "n_series_intersection": int(len(tr_series & va_series)),
                "n_missing_study_uid": int(n_missing_study),
                "n_missing_series_uid": int(n_missing_series),
                "n_fallback_to_series": int(n_fallback_to_series),
                "n_fallback_to_image": int(n_fallback_to_image),
                "split_groups_total": int(split_stats.get("split_groups_total", -1)),
                "split_groups_val": int(split_stats.get("split_groups_val", -1)),
            }
        )

    summary = {
        "hypothesis": "Train/val leakage exists under split_by=study.",
        "rejection_criteria": [
            "n_image_intersection == 0 for all seeds",
            "n_study_intersection == 0 for all seeds",
            "n_series_intersection == 0 for all seeds",
            "n_fallback_to_image == 0 for all seeds",
        ],
        "results": {
            "all_image_intersection_zero": all(r["n_image_intersection"] == 0 for r in rows),
            "all_study_intersection_zero": all(r["n_study_intersection"] == 0 for r in rows),
            "all_series_intersection_zero": all(r["n_series_intersection"] == 0 for r in rows),
            "all_fallback_to_image_zero": all(r["n_fallback_to_image"] == 0 for r in rows),
            "max_missing_study_uid": max(r["n_missing_study_uid"] for r in rows),
            "max_missing_series_uid": max(r["n_missing_series_uid"] for r in rows),
            "max_fallback_to_series": max(r["n_fallback_to_series"] for r in rows),
            "max_fallback_to_image": max(r["n_fallback_to_image"] for r in rows),
        },
        "rows": rows,
    }

    out = Path(str(ns.out_json)).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
