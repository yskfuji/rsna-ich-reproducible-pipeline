from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
from collections.abc import Iterable
import os
import sqlite3
import struct
import zlib

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset


RSNA_CLASSES = (
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
)


_SQLITE_MAGIC = b"RSNP"  # must match tools/precompute_rsna_preprocessed_sqlite.py
_SQLITE_VERSION = 1
_DTYPE_U8 = 1
_DTYPE_F16 = 2
_DTYPE_F32 = 3


_SQLITE_CONNS: dict[str, sqlite3.Connection] = {}


def _sqlite_conn(path: Path) -> sqlite3.Connection:
    p = str(Path(path).expanduser().resolve())
    c = _SQLITE_CONNS.get(p)
    if c is not None:
        return c
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    _SQLITE_CONNS[p] = conn
    return conn


def _deserialize_chw(blob: bytes) -> NDArray[np.float32]:
    if len(blob) < 12:
        raise ValueError("invalid blob (too small)")
    magic, ver, dtype_code, comp, c, h, w = struct.unpack("<4sBBBBHH", blob[:12])
    if magic != _SQLITE_MAGIC or int(ver) != int(_SQLITE_VERSION):
        raise ValueError("invalid blob header")
    payload = blob[12:]
    if int(comp) == 1:
        payload = zlib.decompress(payload)

    if int(dtype_code) == _DTYPE_U8:
        dt = np.uint8
        scale = True
    elif int(dtype_code) == _DTYPE_F16:
        dt = np.float16
        scale = False
    elif int(dtype_code) == _DTYPE_F32:
        dt = np.float32
        scale = False
    else:
        raise ValueError("unknown dtype_code")

    x = np.frombuffer(payload, dtype=dt)
    x = x.reshape((int(c), int(h), int(w)))
    if scale:
        return (x.astype(np.float32) / 255.0).astype(np.float32)
    return x.astype(np.float32)


def read_rsna_preprocessed_to_tensor2d(
    *,
    image_id: str,
    db_path: Path,
) -> torch.Tensor:
    """Read precomputed (C,H,W) float32 tensor from SQLite DB."""
    conn = _sqlite_conn(db_path)
    row = conn.execute("SELECT blob FROM tensors WHERE key=? LIMIT 1;", (str(image_id),)).fetchone()
    if row is None:
        raise FileNotFoundError(f"missing key in preprocessed db: {image_id}")
    blob = row[0]
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError("invalid blob type")
    b = blob.tobytes() if isinstance(blob, memoryview) else bytes(blob)
    arr = _deserialize_chw(b)
    return torch.from_numpy(arr)


def preprocessed_db_has_meta(db_path: Path) -> bool:
    conn = _sqlite_conn(db_path)
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta' LIMIT 1;"
    ).fetchone()
    return row is not None


def preprocessed_db_existing_keys(image_ids: list[str], *, db_path: Path, chunk_size: int = 900) -> set[str]:
    """Return subset of image_ids that exist in tensors table.

    Uses chunked IN queries to avoid 1 query per id.
    """
    ids = [str(x) for x in image_ids if str(x).strip()]
    if not ids:
        return set()
    conn = _sqlite_conn(db_path)
    out: set[str] = set()
    cs = int(chunk_size)
    if cs <= 0:
        cs = 900
    for i in range(0, len(ids), cs):
        chunk = ids[i : i + cs]
        placeholders = ",".join(["?"] * len(chunk))
        q = f"SELECT key FROM tensors WHERE key IN ({placeholders});"
        rows = conn.execute(q, tuple(chunk)).fetchall()
        out.update(str(r[0]) for r in rows if r and r[0] is not None)
    return out


def read_rsna_preprocessed_meta(
    *,
    image_id: str,
    db_path: Path,

) -> tuple[str | None, str | None, int | None, int | None]:
    """Read (study_uid, series_uid, instance_number, series_index) from SQLite meta table.

    Returns (None, None, None) if missing.
    """
    conn = _sqlite_conn(db_path)
    try:
        try:
            row = conn.execute(
                "SELECT study_uid, series_uid, instance_number, series_index FROM meta WHERE key=? LIMIT 1;",
                (str(image_id),),
            ).fetchone()
        except sqlite3.OperationalError:
            # Backward-compatible: older DBs don't have series_index.
            row = conn.execute(
                "SELECT study_uid, series_uid, instance_number FROM meta WHERE key=? LIMIT 1;",
                (str(image_id),),
            ).fetchone()
            if row is not None:
                row = (row[0], row[1], row[2], None)
    except sqlite3.OperationalError:
        return None, None, None, None
    if row is None:
        return None, None, None, None
    study_uid = str(row[0]).strip() if row[0] is not None else None
    series_uid = str(row[1]).strip() if row[1] is not None else None
    inst_i: int | None
    try:
        inst_i = int(row[2]) if row[2] is not None else None
    except Exception:
        inst_i = None
    series_idx: int | None
    try:
        series_idx = int(row[3]) if row[3] is not None else None
    except Exception:
        series_idx = None
    return (study_uid or None), (series_uid or None), inst_i, series_idx


def _read_rsna_preprocessed_by_series_instance(
    *,
    series_uid: str,
    instance_number: int,
    db_path: Path,
) -> torch.Tensor | None:
    conn = _sqlite_conn(db_path)
    try:
        row = conn.execute(
            "SELECT t.blob "
            "FROM meta m JOIN tensors t ON t.key=m.key "
            "WHERE m.series_uid=? AND m.instance_number=? "
            "ORDER BY m.key LIMIT 1;",
            (str(series_uid), int(instance_number)),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    blob = row[0]
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return None
    b = blob.tobytes() if isinstance(blob, memoryview) else bytes(blob)
    arr = _deserialize_chw(b)
    return torch.from_numpy(arr)


def _read_rsna_preprocessed_by_series_index(
    *,
    series_uid: str,
    series_index: int,
    db_path: Path,
) -> torch.Tensor | None:
    conn = _sqlite_conn(db_path)
    try:
        row = conn.execute(
            "SELECT t.blob "
            "FROM meta m JOIN tensors t ON t.key=m.key "
            "WHERE m.series_uid=? AND m.series_index=? "
            "ORDER BY m.key LIMIT 1;",
            (str(series_uid), int(series_index)),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    blob = row[0]
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return None
    b = blob.tobytes() if isinstance(blob, memoryview) else bytes(blob)
    arr = _deserialize_chw(b)
    return torch.from_numpy(arr)


def read_rsna_preprocessed_to_tensor25d(
    *,
    image_id: str,
    db_path: Path,
    stack_slices: int,
) -> torch.Tensor:
    """Read DB-only pseudo-2.5D tensor shaped (C*stack_slices, H, W)."""
    if int(stack_slices) <= 1 or int(stack_slices) % 2 != 1:
        raise ValueError("stack_slices must be odd and >= 3")

    center = read_rsna_preprocessed_to_tensor2d(image_id=image_id, db_path=db_path)
    _, series_uid, inst, series_idx = read_rsna_preprocessed_meta(image_id=image_id, db_path=db_path)
    if series_uid is None:
        return torch.cat([center] * int(stack_slices), dim=0)

    half = int(stack_slices) // 2
    xs: list[torch.Tensor] = []
    for di in range(-half, half + 1):
        if series_idx is not None:
            t = _read_rsna_preprocessed_by_series_index(
                series_uid=series_uid,
                series_index=int(series_idx) + int(di),
                db_path=db_path,
            )
        elif inst is not None:
            t = _read_rsna_preprocessed_by_series_instance(
                series_uid=series_uid,
                instance_number=int(inst) + int(di),
                db_path=db_path,
            )
        else:
            t = None
        xs.append(center if t is None else t)
    return torch.cat(xs, dim=0)


@dataclass(frozen=True)
class RsnaSliceRecord:
    image_id: str  # e.g. ID_12cadc6af
    dcm_path: Path
    y: NDArray[np.float32]  # shape (6,), float32


def compute_subset_fingerprint_sha256(image_ids: Iterable[str]) -> str:
    """Compute a stable fingerprint for a subset defined by image_id list.

    Fingerprint is SHA256 over the UTF-8 bytes of newline-joined, sorted unique
    non-empty image_id strings.
    """

    ids = sorted({str(x).strip() for x in image_ids if str(x).strip()})
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deduplicate_records_by_preprocessed_tensor_hash(
    records: list[RsnaSliceRecord],
    *,
    db_path: Path,
) -> tuple[list[RsnaSliceRecord], dict[str, int]]:
    """Deduplicate records by SHA256 of decoded preprocessed tensor bytes.

    Keeps first occurrence order-stably. Intended to run BEFORE split to prevent
    exact tensor duplicates from being split across train/val.
    """

    conn = _sqlite_conn(db_path)
    kept: list[RsnaSliceRecord] = []
    seen: set[str] = set()
    n_missing_blob = 0
    n_dropped_dup = 0

    for r in records:
        row = conn.execute("SELECT blob FROM tensors WHERE key=? LIMIT 1;", (str(r.image_id),)).fetchone()
        if row is None:
            n_missing_blob += 1
            kept.append(r)
            continue
        blob = row[0]
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            n_missing_blob += 1
            kept.append(r)
            continue
        b = blob.tobytes() if isinstance(blob, memoryview) else bytes(blob)
        arr = _deserialize_chw(b)
        h = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
        if h in seen:
            n_dropped_dup += 1
            continue
        seen.add(h)
        kept.append(r)

    stats = {
        "dedup_records_before": int(len(records)),
        "dedup_records_after": int(len(kept)),
        "dedup_dropped_duplicates": int(n_dropped_dup),
        "dedup_missing_blob": int(n_missing_blob),
    }
    return kept, stats


def _parse_windows(windows: str) -> list[tuple[float, float]]:
    """Parse "L,W;L,W;..." into list[(L,W)]."""
    out: list[tuple[float, float]] = []
    s = str(windows).strip()
    if not s:
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        l_s, w_s = part.split(",")
        out.append((float(l_s), float(w_s)))
    return out


def _window_hu_to_01(hu: NDArray[np.floating[Any]], level: float, width: float) -> NDArray[np.float32]:
    lo = level - width / 2.0
    hi = level + width / 2.0
    x = (hu.astype(np.float32) - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def read_rsna_dicom_to_tensor(
    dcm_path: Path,
    out_size: int = 256,
    windows: str = "40,80;80,200;600,2800",
    preprocess: str = "legacy",
    cache_dir: Path | None = None,
    cache_key: str | None = None,
) -> torch.Tensor:
    """Read a RSNA CT slice DICOM and return tensor shaped (C, D=1, H, W)."""
    import pydicom  # lazy import

    def _fix_12bit_unsigned_wrap(ds: Any, img_i16: NDArray[np.int16]) -> NDArray[np.int16]:
        """Handle rare 12-bit unsigned wrap cases (defensive heuristic).

        Mirrors a common Kaggle RSNA ICH workaround. Only triggers on a narrow
        condition to avoid over-correcting.
        """
        try:
            # Default for PixelRepresentation is 0 (unsigned) when tag is missing,
            # consistent with Kaggle RSNA workarounds for 12-bit unsigned wrap.
            if int(getattr(ds, "BitsStored", 0) or 0) == 12 and int(getattr(ds, "PixelRepresentation", 0) or 0) == 0:
                intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
                if intercept > -100:
                    x = img_i16.astype(np.int32) + 1000
                    x[x >= 4096] = x[x >= 4096] - 4096
                    img_i16 = x.astype(np.int16)
                    try:
                        ds.RescaleIntercept = -1000
                    except Exception:
                        pass
        except Exception:
            return img_i16
        return img_i16

    def _skull_strip_and_crop(
        windowed_chw: NDArray[np.float32],
        hu: NDArray[np.float32],
        *,
        bone_hu_threshold: int = 200,
        morph_close_kernel: int = 0,
        pad_to_square: bool = True,
    ) -> NDArray[np.float32]:
        """Optional masking + crop using a simple HU threshold.

        - Mask keeps pixels with HU < bone_hu_threshold.
        - Optional morphology close (approx) using torch pooling (no extra deps).
        - Crop to bbox of mask; optionally pad to square.
        """
        mask = (hu < float(bone_hu_threshold)).astype(np.float32)  # (H,W)
        if int(morph_close_kernel) and int(morph_close_kernel) > 1:
            k = int(morph_close_kernel)
            # torch-based close: dilate then erode
            mt = torch.from_numpy(mask).view(1, 1, mask.shape[0], mask.shape[1])
            mt = torch.nn.functional.max_pool2d(mt, kernel_size=k, stride=1, padding=k // 2)
            mt = 1.0 - torch.nn.functional.max_pool2d(1.0 - mt, kernel_size=k, stride=1, padding=k // 2)
            mask = mt.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

        # Apply mask
        out = windowed_chw.copy()
        out[:, mask <= 0.0] = 0.0

        ys, xs = np.where(mask > 0.0)
        if ys.size == 0 or xs.size == 0:
            return out

        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        if not bool(pad_to_square):
            return out[:, y1 : y2 + 1, x1 : x2 + 1]

        h, w = mask.shape
        box_h = y2 - y1 + 1
        box_w = x2 - x1 + 1
        size = int(max(box_h, box_w))
        cy = int((y1 + y2) // 2)
        cx = int((x1 + x2) // 2)
        half = int(size // 2)
        yy1 = max(0, cy - half)
        yy2 = min(h, cy + half + 1)
        xx1 = max(0, cx - half)
        xx2 = min(w, cx + half + 1)
        return out[:, yy1:yy2, xx1:xx2]

    def _cache_path() -> Path | None:
        if cache_dir is None:
            return None
        cd = Path(cache_dir).expanduser().resolve()
        try:
            cd.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        pp = str(preprocess).strip().lower()
        # Keep legacy cache naming stable to avoid re-caching by default.
        if pp == "legacy":
            ws_sig = zlib.crc32(str(windows).encode("utf-8")) & 0xFFFFFFFF
        else:
            sig_src = f"{str(windows)}|pp={pp}".encode("utf-8")
            ws_sig = zlib.crc32(sig_src) & 0xFFFFFFFF
        k = str(cache_key).strip() if cache_key is not None else ""
        if not k:
            # Fallback to file stem; not always unique across RSNA, but better than nothing.
            k = Path(dcm_path).stem
        fname = f"{k}_s{int(out_size)}_w{ws_sig}.pt"
        return cd / fname

    def _ensure_transfer_syntax(dataset: Any) -> None:
        try:
            fm = getattr(dataset, "file_meta", None)
            if fm is None:
                from pydicom.dataset import FileMetaDataset

                dataset.file_meta = FileMetaDataset()
                fm = dataset.file_meta
            if getattr(fm, "TransferSyntaxUID", None) is None:
                from pydicom.uid import ImplicitVRLittleEndian

                fm.TransferSyntaxUID = ImplicitVRLittleEndian
        except Exception:
            return

    ws = _parse_windows(windows)
    c = max(1, len(ws)) if str(windows).strip() else 1
    s = int(out_size) if int(out_size) > 0 else 256

    cp = _cache_path()
    if cp is not None and cp.exists():
        try:
            t = torch.load(str(cp), map_location="cpu")
            if isinstance(t, torch.Tensor) and tuple(t.shape) == (c, 1, s, s):
                return t.to(dtype=torch.float32)
        except Exception:
            pass

    pp = str(preprocess).strip().lower()
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        _ensure_transfer_syntax(ds)
        px_i16 = ds.pixel_array.astype(np.int16)
    except Exception:
        # Some RSNA files are missing a DICOM preamble; force=False will raise.
        # If decoding fails, fall back to a tiny constant image to keep training running.
        try:
            ds = pydicom.dcmread(str(dcm_path), force=False)
            _ensure_transfer_syntax(ds)
            px_i16 = ds.pixel_array.astype(np.int16)
        except Exception:
            return torch.full((c, 1, s, s), 1e-4, dtype=torch.float32)

    # Extra DICOM fixes (from researched spec) only in non-legacy modes.
    if pp != "legacy":
        try:
            if str(getattr(ds, "PhotometricInterpretation", "") or "") == "MONOCHROME1":
                px_i16 = (np.max(px_i16) - px_i16).astype(np.int16)
        except Exception:
            pass
        px_i16 = _fix_12bit_unsigned_wrap(ds, px_i16)

    px = px_i16.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    hu = px * slope + intercept

    ws = _parse_windows(windows)
    if not ws:
        # fallback: raw HU clipped to [-1000, 1000] then scaled to [0,1]
        x = np.clip(hu, -1000.0, 1000.0)
        x = (x + 1000.0) / 2000.0
        ch = [x]
    else:
        ch = [_window_hu_to_01(hu, level=l, width=w) for (l, w) in ws]

    x = np.stack(ch, axis=0)  # (C,H,W)
    if pp == "gpt52":
        # Research-backed optional steps: simple skull-strip + crop.
        x = _skull_strip_and_crop(
            x,
            hu,
            bone_hu_threshold=200,
            morph_close_kernel=5,
            pad_to_square=True,
        )
    t = torch.from_numpy(x).unsqueeze(1)  # (C,1,H,W)

    if int(out_size) > 0:
        # Resize H,W while keeping D=1.
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0),
            size=(1, int(out_size), int(out_size)),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

    # Best-effort cache write (atomic replace).
    if cp is not None:
        tmp: Path | None = None
        try:
            tmp = cp.with_suffix(cp.suffix + f".tmp.{os.getpid()}")
            torch.save(t.detach().cpu(), str(tmp))
            os.replace(str(tmp), str(cp))
        except Exception:
            try:
                if tmp is not None and tmp.exists():
                    tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    return t


def read_rsna_dicom_to_tensor2d(
    dcm_path: Path,
    out_size: int = 256,
    windows: str = "40,80;80,200;600,2800",
    preprocess: str = "legacy",
    cache_dir: Path | None = None,
    cache_key: str | None = None,
) -> torch.Tensor:
    """Read a RSNA CT slice DICOM and return tensor shaped (C, H, W)."""
    t = read_rsna_dicom_to_tensor(
        dcm_path=dcm_path,
        out_size=out_size,
        windows=windows,
        preprocess=preprocess,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    return t.squeeze(1)


class RsnaIchSliceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[RsnaSliceRecord],
        out_size: int = 256,
        windows: str = "40,80;80,200;600,2800",
        preprocess: str = "legacy",
        cache_dir: Path | None = None,
        preprocessed_db: Path | None = None,
    ):
        self.records = list(records)
        self.out_size = int(out_size)
        self.windows = str(windows)
        self.preprocess = str(preprocess)
        self.cache_dir = cache_dir
        self.preprocessed_db = preprocessed_db

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        if self.preprocessed_db is not None:
            x2d = read_rsna_preprocessed_to_tensor2d(image_id=r.image_id, db_path=self.preprocessed_db)
            x = x2d.unsqueeze(1)  # (C,1,H,W)
        else:
            x = read_rsna_dicom_to_tensor(
                r.dcm_path,
                out_size=self.out_size,
                windows=self.windows,
                preprocess=self.preprocess,
                cache_dir=self.cache_dir,
                cache_key=r.image_id,
            )
        y = torch.from_numpy(r.y.astype(np.float32))
        return {"image_id": r.image_id, "x": x, "y": y}


class RsnaIchSlice2DDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[RsnaSliceRecord],
        out_size: int = 256,
        windows: str = "40,80;80,200;600,2800",
        preprocess: str = "legacy",
        cache_dir: Path | None = None,
        preprocessed_db: Path | None = None,
    ):
        self.records = list(records)
        self.out_size = int(out_size)
        self.windows = str(windows)
        self.preprocess = str(preprocess)
        self.cache_dir = cache_dir
        self.preprocessed_db = preprocessed_db

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        if self.preprocessed_db is not None:
            x = read_rsna_preprocessed_to_tensor2d(image_id=r.image_id, db_path=self.preprocessed_db)
        else:
            x = read_rsna_dicom_to_tensor2d(
                r.dcm_path,
                out_size=self.out_size,
                windows=self.windows,
                preprocess=self.preprocess,
                cache_dir=self.cache_dir,
                cache_key=r.image_id,
            )
        y = torch.from_numpy(r.y.astype(np.float32))
        return {"image_id": r.image_id, "x": x, "y": y}


def _read_series_and_instance(dcm_path: Path) -> tuple[str, int] | None:
    """Read minimal metadata needed for 2.5D neighboring-slice lookup."""
    import pydicom  # lazy import
    import warnings

    warnings.filterwarnings("ignore", message=r"Invalid value for VR UI:.*", module=r"pydicom\..*")

    try:
        ds = pydicom.dcmread(
            str(dcm_path),
            stop_before_pixels=True,
            specific_tags=["SeriesInstanceUID", "InstanceNumber"],
            force=True,
        )
    except Exception:
        return None

    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
    if not series_uid:
        return None
    inst = getattr(ds, "InstanceNumber", None)
    try:
        inst_i = int(inst) if inst is not None else -1
    except Exception:
        inst_i = -1
    return series_uid, inst_i


class RsnaIchSlice25DDataset(Dataset[dict[str, Any]]):
    """2.5D dataset: stack neighboring slices along channel dim.

    Input shape: (C * stack_slices, H, W)
    """

    def __init__(
        self,
        records: list[RsnaSliceRecord],
        out_size: int = 256,
        windows: str = "40,80;80,200;600,2800",
        stack_slices: int = 3,
        preprocess: str = "legacy",
        cache_dir: Path | None = None,
        preprocessed_db: Path | None = None,
    ):
        if int(stack_slices) <= 1 or int(stack_slices) % 2 != 1:
            raise ValueError("stack_slices must be odd and >= 3")

        self.records = list(records)
        self.out_size = int(out_size)
        self.windows = str(windows)
        self.stack_slices = int(stack_slices)
        self.preprocess = str(preprocess)
        self.cache_dir = cache_dir
        self.preprocessed_db = preprocessed_db

        # DICOM path lookup is only needed for non-preprocessed mode.
        self._meta: dict[str, tuple[str, int] | None] = {}
        self._series_index: dict[str, dict[int, Path]] = {}
        if self.preprocessed_db is None:
            for r in self.records:
                m = _read_series_and_instance(r.dcm_path)
                self._meta[r.image_id] = m
                if m is None:
                    continue
                series_uid, inst = m
                self._series_index.setdefault(series_uid, {})[inst] = r.dcm_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        if self.preprocessed_db is not None:
            x = read_rsna_preprocessed_to_tensor25d(
                image_id=r.image_id,
                db_path=self.preprocessed_db,
                stack_slices=int(self.stack_slices),
            )
        else:
            m = self._meta.get(r.image_id)
            if m is None:
                # Fallback: replicate center slice
                x0 = read_rsna_dicom_to_tensor2d(
                    r.dcm_path,
                    out_size=self.out_size,
                    windows=self.windows,
                    preprocess=self.preprocess,
                    cache_dir=self.cache_dir,
                    cache_key=r.image_id,
                )
                xs = [x0] * self.stack_slices
            else:
                series_uid, inst = m
                series = self._series_index.get(series_uid, {})
                half = self.stack_slices // 2
                xs = []
                for di in range(-half, half + 1):
                    p = series.get(inst + di)
                    if p is None:
                        p = r.dcm_path
                    xs.append(
                        read_rsna_dicom_to_tensor2d(
                            p,
                            out_size=self.out_size,
                            windows=self.windows,
                            preprocess=self.preprocess,
                            cache_dir=self.cache_dir,
                            cache_key=p.stem,
                        )
                    )
            x = torch.cat(xs, dim=0)  # (C*stack_slices,H,W)
        y = torch.from_numpy(r.y.astype(np.float32))
        return {"image_id": r.image_id, "x": x, "y": y}


def iter_rsna_stage2_records_from_csv(
    csv_path: Path,
    dicom_dir: Path,
    limit_images: int | None = 5000,
    seed: int = 0,
) -> list[RsnaSliceRecord]:
    """Read `stage_2_train.csv` and build per-slice records.

    Notes:
    - `stage_2_train.csv` is very large.
    - When `limit_images` is not None, this function performs a 2-pass read:
      1) deterministically sample `limit_images` unique `image_id`s uniformly across
         the entire CSV (to avoid bias from the head of the file)
      2) read labels for the sampled `image_id`s and ensure all 6 class labels exist
    - Output labels are in RSNA_CLASSES order.
    """
    import hashlib
    import heapq

    import pandas as pd

    csv_path = Path(csv_path)
    dicom_dir = Path(dicom_dir)

    def _stable_key_u64(image_id: str) -> int:
        # Stable across runs/platforms; avoids Python's randomized hash.
        b = hashlib.blake2b(f"{int(seed)}:{image_id}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(b, byteorder="big", signed=False)

    # Pass-1: choose image_ids (uniform-ish via smallest random keys).
    target = None if limit_images is None else int(limit_images)
    selected_key: dict[str, int] | None = None
    selected_ids: set[str] | None = None

    if target is not None:
        if target <= 0:
            return []
        selected_key = {}
        # Keep max-heap of size K using (-key, image_id)
        heap: list[tuple[int, str]] = []
        for chunk in pd.read_csv(csv_path, usecols=("ID",), chunksize=200_000):  # type: ignore[call-overload]
            ids = chunk["ID"].astype(str)
            image_ids = ids.str.rsplit("_", n=1).str[0].to_numpy()
            for img in image_ids:
                if img in selected_key:
                    continue
                k = _stable_key_u64(str(img))
                if len(heap) < target:
                    selected_key[str(img)] = k
                    heapq.heappush(heap, (-k, str(img)))
                    continue

                # If k is smaller than the current max key in reservoir, replace it.
                max_k = -heap[0][0]
                if k < max_k:
                    _, removed = heapq.heapreplace(heap, (-k, str(img)))
                    try:
                        del selected_key[removed]
                    except KeyError:
                        pass
                    selected_key[str(img)] = k

        selected_ids = set(selected_key.keys())

    # Pass-2: read labels (either for all ids or only selected ids).
    labels: dict[str, NDArray[np.float32]] = {}
    seen_mask: dict[str, int] = {}
    all_bits = (1 << len(RSNA_CLASSES)) - 1

    for chunk in pd.read_csv(csv_path, usecols=("ID", "Label"), chunksize=200_000):  # type: ignore[call-overload]
        ids = chunk["ID"].astype(str)
        y = chunk["Label"].astype(np.float32).to_numpy()

        image_id = ids.str.rsplit("_", n=1).str[0]
        cls = ids.str.rsplit("_", n=1).str[1]

        for img, c, v in zip(image_id.to_numpy(), cls.to_numpy(), y, strict=False):
            img_s = str(img)
            if selected_ids is not None and img_s not in selected_ids:
                continue
            try:
                k = RSNA_CLASSES.index(str(c))
            except ValueError:
                continue
            if img_s not in labels:
                labels[img_s] = np.zeros((len(RSNA_CLASSES),), dtype=np.float32)
                seen_mask[img_s] = 0
            labels[img_s][k] = float(v)
            seen_mask[img_s] |= (1 << int(k))

    # Ensure full 6-label completeness when sampling.
    if selected_ids is not None:
        incomplete = [img for img in list(labels.keys()) if int(seen_mask.get(img, 0)) != all_bits]
        for img in incomplete:
            labels.pop(img, None)
            seen_mask.pop(img, None)
        if selected_key is not None:
            for img in incomplete:
                selected_key.pop(img, None)

    # Deterministic ordering
    if selected_key is not None:
        keys = sorted(labels.keys(), key=lambda k: int(selected_key.get(k, 0)))
    else:
        rng = np.random.default_rng(int(seed))
        keys = list(labels.keys())
        rng.shuffle(keys)

    out: list[RsnaSliceRecord] = []
    for img in keys:
        p = dicom_dir / f"{img}.dcm"
        out.append(RsnaSliceRecord(image_id=img, dcm_path=p, y=labels[img]))

    return out
