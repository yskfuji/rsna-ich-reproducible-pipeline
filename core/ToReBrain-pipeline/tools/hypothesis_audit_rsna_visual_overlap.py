#!/usr/bin/env python
from __future__ import annotations

import argparse
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


def _sqlite_conn(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path), check_same_thread=False)


def _deserialize_chw(blob: bytes) -> np.ndarray:
    if len(blob) < 12:
        raise ValueError("invalid blob")
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


def _read_blob_to_gray(conn: sqlite3.Connection, image_id: str) -> np.ndarray:
    row = conn.execute("SELECT blob FROM tensors WHERE key=? LIMIT 1;", (str(image_id),)).fetchone()
    if row is None:
        raise KeyError(image_id)
    blob = row[0]
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    arr = _deserialize_chw(bytes(blob))
    # Mean across channels -> grayscale-like representation.
    g = arr.mean(axis=0)
    return g


def _ahash64(gray_hw: np.ndarray) -> int:
    # downsample to 8x8 via simple stride-friendly averaging
    h, w = gray_hw.shape
    ys = np.linspace(0, h, 9, dtype=int)
    xs = np.linspace(0, w, 9, dtype=int)
    small = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            patch = gray_hw[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            small[i, j] = float(np.mean(patch)) if patch.size else 0.0
    m = float(np.mean(small))
    bits = (small >= m).astype(np.uint8).reshape(-1)
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return int(v)


def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Audit leakage hypothesis via visual-near-duplicate checks across train/val, "
            "in addition to study/series disjointness."
        )
    )
    p.add_argument("--rsna-root", required=True, type=str)
    p.add_argument("--preprocessed-root", required=True, type=str)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)
    p.add_argument("--limit-images", default=8000, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)
    p.add_argument("--hamming-threshold", default=4, type=int, help="Near-duplicate threshold for 64-bit ahash")
    p.add_argument("--out-json", default="/tmp/rsna_visual_overlap_audit.json", type=str)
    ns = p.parse_args(argv)

    seeds = [int(x.strip()) for x in str(ns.seeds).split(",") if x.strip()]
    rsna_root = Path(str(ns.rsna_root)).expanduser().resolve()
    pre_root = Path(str(ns.preprocessed_root)).expanduser().resolve()
    pre_db = pre_root / "train.sqlite"

    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    conn = _sqlite_conn(pre_db)

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

        tr_hashes: dict[int, int] = {}
        va_hashes: dict[int, int] = {}

        for img in tr_ids:
            g = _read_blob_to_gray(conn, img)
            h = _ahash64(g)
            tr_hashes[h] = tr_hashes.get(h, 0) + 1

        for img in va_ids:
            g = _read_blob_to_gray(conn, img)
            h = _ahash64(g)
            va_hashes[h] = va_hashes.get(h, 0) + 1

        exact_overlap_hashes = set(tr_hashes.keys()) & set(va_hashes.keys())
        n_exact_hash_overlap = len(exact_overlap_hashes)

        # near-duplicate count at hash level (not pairwise image count)
        tr_keys = list(tr_hashes.keys())
        va_keys = list(va_hashes.keys())
        near_cnt = 0
        thr = int(ns.hamming_threshold)
        for vh in va_keys:
            found = False
            for th in tr_keys:
                if _hamming64(vh, th) <= thr:
                    found = True
                    break
            if found:
                near_cnt += 1

        rows.append(
            {
                "seed": int(seed),
                "n_train": int(len(tr_ids)),
                "n_val": int(len(va_ids)),
                "n_unique_train_hash": int(len(tr_hashes)),
                "n_unique_val_hash": int(len(va_hashes)),
                "n_exact_hash_overlap": int(n_exact_hash_overlap),
                "n_val_hashes_with_near_match_in_train": int(near_cnt),
                "near_match_ratio_on_val_hash": float(near_cnt / max(1, len(va_hashes))),
            }
        )

    summary = {
        "hypothesis": "Leakage exists via same/similar images across train/val.",
        "method": "8x8 aHash on preprocessed tensors (channel-mean), exact overlap and Hamming-near overlap.",
        "hamming_threshold": int(ns.hamming_threshold),
        "results": {
            "max_exact_hash_overlap": int(max(r["n_exact_hash_overlap"] for r in rows)),
            "max_near_match_ratio_on_val_hash": float(max(r["near_match_ratio_on_val_hash"] for r in rows)),
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
