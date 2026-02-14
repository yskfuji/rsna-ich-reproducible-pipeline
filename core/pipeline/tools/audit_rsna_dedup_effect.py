#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.datasets.rsna_ich_dataset import (  # noqa: E402
    RSNA_CLASSES,
    deduplicate_records_by_preprocessed_tensor_hash,
    iter_rsna_stage2_records_from_csv,
    preprocessed_db_existing_keys,
    read_rsna_preprocessed_meta,
)
from src.training.train_rsna_cnn2d_classifier import _group_split_records  # noqa: E402

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


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    rsna_root = repo / "Datasets" / "rsna_preprocessed_gpt52_img384_w3_f32" / "rsna_meta"
    pre_db = repo / "Datasets" / "rsna_preprocessed_gpt52_img384_w3_f32" / "train.sqlite"
    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    conn = sqlite3.connect(str(pre_db), check_same_thread=False)

    any_idx = int(RSNA_CLASSES.index("any"))

    def _pos_group_selector(records):
        return any(float(r.y[any_idx]) > 0.5 for r in records)

    rows = []
    for seed in range(10):
        records = iter_rsna_stage2_records_from_csv(
            csv_path=csv_path,
            dicom_dir=dcm_dir,
            limit_images=8000,
            seed=seed,
        )
        exist = preprocessed_db_existing_keys([str(r.image_id) for r in records], db_path=pre_db)
        records = [r for r in records if str(r.image_id) in exist]
        records, dedup_stats = deduplicate_records_by_preprocessed_tensor_hash(records, db_path=pre_db)

        group_ids = []
        for r in records:
            st, se, _i, _s = read_rsna_preprocessed_meta(image_id=str(r.image_id), db_path=pre_db)
            group_ids.append(st or se or str(r.image_id))

        tr, va, _ = _group_split_records(
            records=records,
            group_ids=group_ids,
            val_frac=0.05,
            seed=seed,
            pos_group_selector=_pos_group_selector,
        )

        tr_h = {_tensor_sha256(conn, str(r.image_id)) for r in tr}
        va_h = {_tensor_sha256(conn, str(r.image_id)) for r in va}

        rows.append(
            {
                "seed": seed,
                "dedup_dropped_duplicates": int(dedup_stats["dedup_dropped_duplicates"]),
                "n_hash_intersection_after_dedup": int(len(tr_h & va_h)),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
            }
        )

    out = {
        "all_zero_after_dedup": all(r["n_hash_intersection_after_dedup"] == 0 for r in rows),
        "max_hash_intersection_after_dedup": max(r["n_hash_intersection_after_dedup"] for r in rows),
        "rows": rows,
    }

    out_path = Path("/tmp/rsna_dedup_before_split_audit.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
