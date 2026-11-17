from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return "sha256:" + h.hexdigest()


def sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(data)


def sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def sha256_file_stat(path: Path) -> str:
    st = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    return sha256_json(payload)


def try_git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    created_utc: str
    git_commit: str
    seed: int
    config_path: str | None
    config_hash: str | None
    dataset_hash: str
    dataset_hash_mode: str


def make_run_id(prefix: str = "diag") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    rand = os.urandom(3).hex()
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in {"-", "_"})
    safe_prefix = safe_prefix or "run"
    return f"{safe_prefix}_{ts}_{pid}_{rand}"


def compute_dataset_hash(
    csv_path: Path,
    root: Path,
    split: str | None = None,
    mode: Literal["stat", "full"] = "stat",
) -> tuple[str, str]:
    """Compute a dataset hash.

    - stat: hashes file path + size + mtime for referenced image/label files (fast).
    - full: hashes file bytes via sha256 (slow).
    """
    import pandas as pd

    df = pd.read_csv(str(csv_path))
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)

    items: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        case_id = str(row["case_id"])

        img_path_raw = row.get("image_path") if "image_path" in df.columns else None
        if isinstance(img_path_raw, str) and img_path_raw.strip():
            img_path = Path(img_path_raw.strip())
            if not img_path.is_absolute():
                img_path = root / img_path
        else:
            img_path = root / "images" / f"{case_id}.nii.gz"

        lbl_path_raw = row.get("label_path") if "label_path" in df.columns else None
        if isinstance(lbl_path_raw, str) and lbl_path_raw.strip():
            lbl_path = Path(lbl_path_raw.strip())
            if not lbl_path.is_absolute():
                lbl_path = root / lbl_path
        else:
            lbl_path = root / "labels" / f"{case_id}.nii.gz"

        if mode == "full":
            from src.preprocess.utils_io import sha256_file

            img_h = sha256_file(str(img_path))
            lbl_h = sha256_file(str(lbl_path)) if lbl_path.exists() else "missing"
        else:
            img_h = sha256_file_stat(img_path)
            lbl_h = sha256_file_stat(lbl_path) if lbl_path.exists() else "missing"

        items.append((case_id + "/image", img_h))
        items.append((case_id + "/label", lbl_h))

    payload = {
        "csv": sha256_file_stat(csv_path) if mode == "stat" else _sha256_bytes(csv_path.read_bytes()),
        "root": str(root.resolve()),
        "split": split,
        "mode": mode,
        "items": items,
    }
    return sha256_json(payload), mode


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def meta_to_dict(meta: RunMeta) -> dict[str, Any]:
    return asdict(meta)
