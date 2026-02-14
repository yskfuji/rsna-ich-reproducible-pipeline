#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.datasets.rsna_ich_dataset import (
    RSNA_CLASSES,
    iter_rsna_stage2_records_from_csv,
    preprocessed_db_existing_keys,
    read_rsna_preprocessed_meta,
)
from src.training.train_rsna_cnn2d_classifier import _group_split_records


_SQLITE_MAGIC = b"RSNP"
_SQLITE_VERSION = 1
_DTYPE_U8 = 1
_DTYPE_F16 = 2
_DTYPE_F32 = 3


def _deserialize_chw(blob: bytes) -> np.ndarray:
    magic, ver, dtype_code, comp, c, h, w = struct.unpack("<4sBBBBHH", blob[:12])
    if magic != _SQLITE_MAGIC or int(ver) != int(_SQLITE_VERSION):
        raise ValueError("invalid header")
    payload = blob[12:]
    if int(comp) == 1:
        payload = zlib.decompress(payload)
    if int(dtype_code) == _DTYPE_U8:
        arr = np.frombuffer(payload, dtype=np.uint8).reshape((int(c), int(h), int(w))).astype(np.float32) / 255.0
    elif int(dtype_code) == _DTYPE_F16:
        arr = np.frombuffer(payload, dtype=np.float16).reshape((int(c), int(h), int(w))).astype(np.float32)
    elif int(dtype_code) == _DTYPE_F32:
        arr = np.frombuffer(payload, dtype=np.float32).reshape((int(c), int(h), int(w))).astype(np.float32)
    else:
        raise ValueError("unknown dtype")
    return arr


def _tensor_sha256(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT blob FROM tensors WHERE key=? LIMIT 1;", (str(key),)).fetchone()
    if row is None:
        raise KeyError(key)
    b = row[0]
    if isinstance(b, memoryview):
        b = b.tobytes()
    arr = _deserialize_chw(bytes(b))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="List exact duplicate tensor hashes across train/val per seed.")
    p.add_argument("--rsna-root", required=True, type=str)
    p.add_argument("--preprocessed-root", required=True, type=str)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)
    p.add_argument("--limit-images", default=8000, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)
    p.add_argument("--out-json", default="/tmp/rsna_duplicate_pairs.json", type=str)
    ns = p.parse_args(argv)

    seeds = [int(x.strip()) for x in str(ns.seeds).split(",") if x.strip()]
    rsna_root = Path(str(ns.rsna_root)).expanduser().resolve()
    pre_db = Path(str(ns.preprocessed_root)).expanduser().resolve() / "train.sqlite"

    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    conn = sqlite3.connect(str(pre_db), check_same_thread=False)

    any_idx = int(RSNA_CLASSES.index("any"))

    def _pos_group_selector(records: list[Any]) -> bool:
        return any(float(r.y[any_idx]) > 0.5 for r in records)

    all_rows = []

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
        for r in records:
            img = str(r.image_id)
            if img not in meta:
                study_uid, series_uid, _inst, _series_idx = read_rsna_preprocessed_meta(image_id=img, db_path=pre_db)
                meta[img] = (study_uid, series_uid)
            st, se = meta[img]
            group_ids.append(st or se or img)

        train_records, val_records, _stats = _group_split_records(
            records=records,
            group_ids=group_ids,
            val_frac=float(ns.val_frac),
            seed=int(seed),
            pos_group_selector=_pos_group_selector,
        )

        tr_map: dict[str, list[str]] = {}
        va_map: dict[str, list[str]] = {}
        for r in train_records:
            img = str(r.image_id)
            h = _tensor_sha256(conn, img)
            tr_map.setdefault(h, []).append(img)
        for r in val_records:
            img = str(r.image_id)
            h = _tensor_sha256(conn, img)
            va_map.setdefault(h, []).append(img)

        inter = sorted(set(tr_map) & set(va_map))
        dup_items = []
        for h in inter:
            tr_ids = tr_map[h]
            va_ids = va_map[h]
            tr_meta = [meta[i] for i in tr_ids[:3]]
            va_meta = [meta[i] for i in va_ids[:3]]
            dup_items.append(
                {
                    "hash": h,
                    "n_train_images": len(tr_ids),
                    "n_val_images": len(va_ids),
                    "train_ids_head": tr_ids[:3],
                    "val_ids_head": va_ids[:3],
                    "train_meta_head": tr_meta,
                    "val_meta_head": va_meta,
                }
            )

        all_rows.append(
            {
                "seed": int(seed),
                "n_duplicate_hashes": len(inter),
                "duplicates": dup_items,
            }
        )

    out = {
        "rows": all_rows,
        "max_duplicate_hashes": max(r["n_duplicate_hashes"] for r in all_rows),
    }

    out_p = Path(str(ns.out_json)).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
