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


def _blob_sha256(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT blob FROM tensors WHERE key=? LIMIT 1;", (str(key),)).fetchone()
    if row is None:
        raise KeyError(key)
    b = row[0]
    if isinstance(b, memoryview):
        b = b.tobytes()
    return hashlib.sha256(bytes(b)).hexdigest()


def _tensor_sha256(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT blob FROM tensors WHERE key=? LIMIT 1;", (str(key),)).fetchone()
    if row is None:
        raise KeyError(key)
    b = row[0]
    if isinstance(b, memoryview):
        b = b.tobytes()
    arr = _deserialize_chw(bytes(b))
    # deterministic bytes from float32 tensor
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strict visual overlap audit using SHA256 hashes.")
    p.add_argument("--rsna-root", required=True, type=str)
    p.add_argument("--preprocessed-root", required=True, type=str)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)
    p.add_argument("--limit-images", default=8000, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)
    p.add_argument("--out-json", default="/tmp/rsna_visual_overlap_audit_strict.json", type=str)
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

        tr_ids = [str(r.image_id) for r in train_records]
        va_ids = [str(r.image_id) for r in val_records]

        tr_blob = {_blob_sha256(conn, i) for i in tr_ids}
        va_blob = {_blob_sha256(conn, i) for i in va_ids}
        tr_tensor = {_tensor_sha256(conn, i) for i in tr_ids}
        va_tensor = {_tensor_sha256(conn, i) for i in va_ids}

        rows.append(
            {
                "seed": int(seed),
                "n_train": int(len(tr_ids)),
                "n_val": int(len(va_ids)),
                "n_blob_sha_intersection": int(len(tr_blob & va_blob)),
                "n_tensor_sha_intersection": int(len(tr_tensor & va_tensor)),
            }
        )

    summary = {
        "method": "Exact duplicate check via SHA256(blob) and SHA256(decoded_tensor)",
        "results": {
            "all_blob_sha_intersection_zero": all(r["n_blob_sha_intersection"] == 0 for r in rows),
            "all_tensor_sha_intersection_zero": all(r["n_tensor_sha_intersection"] == 0 for r in rows),
            "max_blob_sha_intersection": max(r["n_blob_sha_intersection"] for r in rows),
            "max_tensor_sha_intersection": max(r["n_tensor_sha_intersection"] for r in rows),
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
