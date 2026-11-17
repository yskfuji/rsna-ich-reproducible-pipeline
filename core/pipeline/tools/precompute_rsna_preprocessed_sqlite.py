#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import struct

import numpy as np
import torch

import pydicom  # type: ignore

# Make `src` importable regardless of current working directory.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Reuse the exact preprocessing implementation used by training/inference.
from src.datasets.rsna_ich_dataset import read_rsna_dicom_to_tensor2d  # noqa: E402


MAGIC = b"RSNP"  # RSNA Preprocessed
VERSION = 1

DTYPE_U8 = 1
DTYPE_F16 = 2
DTYPE_F32 = 3


@dataclass(frozen=True)
class PreprocessJob:
    key: str
    dcm_path: Path


def _resolve_rsna_root(maybe: str | None) -> Path:
    v = (maybe or os.environ.get("RSNA_ROOT") or "").strip()
    if not v:
        raise SystemExit("Missing RSNA root. Provide --rsna-root or set env RSNA_ROOT.")
    return Path(v).expanduser().resolve()


def _iter_dicom_jobs(dicom_dir: Path) -> Iterator[PreprocessJob]:
    # Faster than glob for huge dirs.
    with os.scandir(dicom_dir) as it:
        for ent in it:
            if not ent.is_file():
                continue
            if not ent.name.lower().endswith(".dcm"):
                continue
            key = Path(ent.name).stem
            yield PreprocessJob(key=key, dcm_path=Path(ent.path))


def _dtype_code(name: str) -> int:
    s = str(name).strip().lower()
    if s in {"u8", "uint8"}:
        return DTYPE_U8
    if s in {"f16", "float16"}:
        return DTYPE_F16
    if s in {"f32", "float32"}:
        return DTYPE_F32
    raise ValueError("dtype must be uint8|float16|float32")


def _serialize_chw(arr: np.ndarray, *, dtype_code: int, compress_level: int) -> bytes:
    if arr.ndim != 3:
        raise ValueError(f"expected (C,H,W), got shape={arr.shape}")

    if dtype_code == DTYPE_U8:
        x = np.clip(arr, 0.0, 1.0)
        x = (x * 255.0 + 0.5).astype(np.uint8)
    elif dtype_code == DTYPE_F16:
        x = arr.astype(np.float16)
    elif dtype_code == DTYPE_F32:
        x = arr.astype(np.float32)
    else:
        raise ValueError(f"unknown dtype_code={dtype_code}")

    x = np.ascontiguousarray(x)
    c, h, w = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))

    raw = x.tobytes(order="C")
    comp = 0
    payload = raw
    if int(compress_level) > 0:
        comp = 1
        payload = zlib.compress(raw, level=int(compress_level))

    header = struct.pack("<4sBBBBHH", MAGIC, VERSION, int(dtype_code), int(comp), int(c), int(h), int(w))
    return header + payload


def _ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB
    conn.execute("CREATE TABLE IF NOT EXISTS tensors (key TEXT PRIMARY KEY, blob BLOB NOT NULL);")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta ("
        "key TEXT PRIMARY KEY, "
        "study_uid TEXT, "
        "series_uid TEXT, "
        "instance_number INTEGER, "
        "position_z REAL, "
        "series_index INTEGER"
        ");"
    )
    # Backward-compatible: add missing columns on existing DBs.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(meta);").fetchall() if row and row[1] is not None}
    if "position_z" not in cols:
        conn.execute("ALTER TABLE meta ADD COLUMN position_z REAL;")
    if "series_index" not in cols:
        conn.execute("ALTER TABLE meta ADD COLUMN series_index INTEGER;")

    # Indexes (create after ALTER TABLE so column-based indexes don't fail).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_study ON meta(study_uid);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_series ON meta(series_uid);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_series_inst ON meta(series_uid, instance_number);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_series_idx ON meta(series_uid, series_index);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_series_z ON meta(series_uid, position_z);")
    return conn


def _key_exists(conn: sqlite3.Connection, key: str) -> bool:
    row = conn.execute("SELECT 1 FROM tensors WHERE key=? LIMIT 1;", (key,)).fetchone()
    return row is not None


def _read_meta(dcm_path: Path) -> tuple[str | None, str | None, int | None, float | None]:
    import warnings

    # RSNA has many non-conformant UI values like "ID_xxx"; silence to avoid huge logs.
    warnings.filterwarnings("ignore", message=r"Invalid value for VR UI:.*")
    try:
        ds = pydicom.dcmread(
            str(dcm_path),
            stop_before_pixels=True,
            specific_tags=["StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber", "ImagePositionPatient"],
            force=True,
        )
    except Exception:
        return None, None, None, None

    study_uid = str(getattr(ds, "StudyInstanceUID", "") or "").strip() or None
    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip() or None
    inst = getattr(ds, "InstanceNumber", None)
    try:
        inst_i = int(inst) if inst is not None else None
    except Exception:
        inst_i = None
    z: float | None = None
    ipp = getattr(ds, "ImagePositionPatient", None)
    try:
        if ipp is not None and len(ipp) >= 3:
            z = float(ipp[2])
    except Exception:
        z = None
    return study_uid, series_uid, inst_i, z


def _backfill_series_index(conn: sqlite3.Connection, *, split_name: str) -> None:
    """Assign contiguous series_index within each series.

    Preference order for sorting:
      1) instance_number (if present)
      2) position_z (if present)
      3) key (fallback)
    """
    series_rows = conn.execute("SELECT DISTINCT series_uid FROM meta WHERE series_uid IS NOT NULL;").fetchall()
    series_uids = [r[0] for r in series_rows if r and r[0] is not None]
    if not series_uids:
        return

    t0 = time.time()
    n_series = 0
    n_keys = 0

    cur = conn.cursor()
    cur.execute("BEGIN;")
    for series_uid in series_uids:
        rows = conn.execute(
            "SELECT key, instance_number, position_z FROM meta WHERE series_uid=?;",
            (str(series_uid),),
        ).fetchall()
        if not rows:
            continue

        def sort_key(r: tuple[object, object, object]) -> tuple[int, float, str]:
            k = str(r[0])
            inst = r[1]
            z = r[2]
            if inst is not None:
                try:
                    return (0, float(int(inst)), k)
                except Exception:
                    pass
            if z is not None:
                try:
                    return (1, float(z), k)
                except Exception:
                    pass
            return (2, 0.0, k)

        rows_s = sorted(rows, key=sort_key)
        updates = [(int(i), str(r[0])) for i, r in enumerate(rows_s)]
        cur.executemany("UPDATE meta SET series_index=? WHERE key=?;", updates)
        n_series += 1
        n_keys += len(updates)
        if (n_series % 2000) == 0:
            dt = max(1e-6, time.time() - t0)
            print(
                f"[{split_name}] backfill series_index series={n_series}/{len(series_uids)} keys={n_keys} elapsed={dt/60:.1f}m",
                flush=True,
            )

    conn.commit()


def _copy_meta_csvs(rsna_root: Path, meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in ["stage_2_train.csv", "stage_2_sample_submission.csv"]:
        src = rsna_root / name
        if src.exists():
            shutil.copy2(src, meta_dir / name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Precompute RSNA preprocessed tensors into a single SQLite DB per split.")
    p.add_argument("--rsna-root", default=None, help="RSNA dataset root (contains stage_2_train/, stage_2_test/, and CSVs)")
    p.add_argument("--out-root", required=True, help="Output root directory (will create train.sqlite/test.sqlite + rsna_meta)")
    p.add_argument("--split", default="both", choices=["train", "test", "both"])
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--windows", type=str, default="40,80;80,200;600,2800")
    p.add_argument("--preprocess", type=str, default="gpt52", help="legacy|gpt52")
    p.add_argument("--dtype", type=str, default="uint8", help="uint8|float16|float32. uint8 is smallest but quantized.")
    p.add_argument("--compress", type=int, default=1, help="zlib compression level (0=none, 1=fast, 6=smaller)")
    p.add_argument("--commit-every", type=int, default=512)
    p.add_argument("--resume", action="store_true", help="Skip keys already present in DB")
    p.add_argument(
        "--meta-only",
        action="store_true",
        help="Only (re)write meta table (Study/Series/Instance) without writing tensor blobs. Useful to backfill meta on an existing DB.",
    )
    p.add_argument("--limit", type=int, default=0, help="For smoke: process only first N dicoms (0=all)")

    ns = p.parse_args(argv)

    rsna_root = _resolve_rsna_root(ns.rsna_root)
    out_root = Path(ns.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    dtype_code = _dtype_code(ns.dtype)

    # Copy CSVs so later training/inference can use a tiny rsna_root even without DICOM.
    _copy_meta_csvs(rsna_root, out_root / "rsna_meta")

    tasks: list[tuple[str, Path, Path]] = []
    if ns.split in {"train", "both"}:
        tasks.append(("train", rsna_root / "stage_2_train", out_root / "train.sqlite"))
    if ns.split in {"test", "both"}:
        tasks.append(("test", rsna_root / "stage_2_test", out_root / "test.sqlite"))

    for split_name, dicom_dir, db_path in tasks:
        if not dicom_dir.exists():
            raise FileNotFoundError(f"Missing DICOM dir: {dicom_dir}")

        conn = _ensure_db(db_path)
        cur = conn.cursor()

        n_done = 0
        n_skipped = 0
        n_meta = 0
        t0 = time.time()
        last_commit = time.time()

        commit_every = int(ns.commit_every)
        limit = int(ns.limit)

        cur.execute("BEGIN;")
        try:
            for i, job in enumerate(_iter_dicom_jobs(dicom_dir), start=1):
                if limit > 0 and i > limit:
                    break

                study_uid, series_uid, inst_i, z = _read_meta(job.dcm_path)
                cur.execute(
                    "INSERT OR REPLACE INTO meta(key, study_uid, series_uid, instance_number, position_z) VALUES(?, ?, ?, ?, ?);",
                    (job.key, study_uid, series_uid, inst_i, z),
                )
                n_meta += 1

                if bool(ns.meta_only):
                    if commit_every > 0 and (n_meta % commit_every) == 0:
                        conn.commit()
                        cur.execute("BEGIN;")
                        last_commit = time.time()
                    if (n_meta % 2000) == 0:
                        dt = max(1e-6, time.time() - t0)
                        rate = n_meta / dt
                        print(
                            f"[{split_name}] meta_only meta={n_meta} scanned={i} rate={rate:.2f}/s elapsed={dt/60:.1f}m db={db_path.name}",
                            flush=True,
                        )
                    continue

                if ns.resume and _key_exists(conn, job.key):
                    n_skipped += 1
                    continue

                t = read_rsna_dicom_to_tensor2d(
                    job.dcm_path,
                    out_size=int(ns.image_size),
                    windows=str(ns.windows),
                    preprocess=str(ns.preprocess),
                    cache_dir=None,
                    cache_key=None,
                )
                # tensor is (C,H,W) float32
                arr = t.detach().cpu().numpy().astype(np.float32)
                blob = _serialize_chw(arr, dtype_code=dtype_code, compress_level=int(ns.compress))

                cur.execute("INSERT OR REPLACE INTO tensors(key, blob) VALUES(?, ?);", (job.key, sqlite3.Binary(blob)))
                n_done += 1

                if commit_every > 0 and (n_done % commit_every) == 0:
                    conn.commit()
                    cur.execute("BEGIN;")
                    last_commit = time.time()

                if (n_done % 2000) == 0:
                    dt = max(1e-6, time.time() - t0)
                    rate = n_done / dt
                    msg = (
                        f"[{split_name}] done={n_done} skipped={n_skipped} scanned={i} "
                        f"rate={rate:.2f}/s elapsed={dt/60:.1f}m db={db_path.name}"
                    )
                    print(msg, flush=True)

            conn.commit()

            # Ensure series_index exists even when InstanceNumber is absent.
            print(f"[{split_name}] backfilling series_index...", flush=True)
            _backfill_series_index(conn, split_name=split_name)
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()

        dt = max(1e-6, time.time() - t0)
        if bool(ns.meta_only):
            print(f"[{split_name}] finished(meta_only): meta={n_meta} elapsed={dt/60:.1f}m -> {db_path}")
        else:
            print(f"[{split_name}] finished: done={n_done} skipped={n_skipped} elapsed={dt/60:.1f}m -> {db_path}")

    print("[done] preprocessed DB(s) created under:", out_root)
    print("[hint] point rsna_root to:", out_root / "rsna_meta")
    print("[hint] point preprocessed_root to:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
