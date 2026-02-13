from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml

from tools.run_meta import (
    RunMeta,
    compute_dataset_hash,
    make_run_id,
    meta_to_dict,
    sha256_json,
    try_git_commit,
    utc_now_iso,
    write_json,
)


def init_or_load_run(
    *,
    repo_root: Path,
    out_root: Path,
    run_id: str | None,
    seed: int,
    config_path: Path | None,
    config_obj: dict[str, Any] | None,
    csv_path: Path,
    data_root: Path,
    dataset_hash_mode: Literal["stat", "full"] = "stat",
) -> tuple[RunMeta, Path]:
    """Create or load `runs/<run_id>/meta.json`.

    If run_id exists, returns existing meta.
    Otherwise creates a new run_id and writes meta.
    """
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = make_run_id("diag")

    run_dir = out_root / run_id
    meta_path = run_dir / "meta.json"

    if meta_path.exists():
        import json

        meta = RunMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
        return meta, run_dir

    config_hash = sha256_json(config_obj) if config_obj is not None else None
    dataset_hash, mode_used = compute_dataset_hash(csv_path, data_root, split=None, mode=dataset_hash_mode)

    meta = RunMeta(
        run_id=run_id,
        created_utc=utc_now_iso(),
        git_commit=try_git_commit(repo_root),
        seed=int(seed),
        config_path=str(config_path) if config_path is not None else None,
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        dataset_hash_mode=str(mode_used),
    )

    write_json(meta_path, meta_to_dict(meta))

    # Also persist config snapshot if provided.
    if config_obj is not None:
        write_json(run_dir / "config_used.json", {"meta": meta_to_dict(meta), "config": config_obj})
    elif config_path is not None and config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text())
            write_json(run_dir / "config_used.json", {"meta": meta_to_dict(meta), "config": cfg})
        except Exception:
            pass

    return meta, run_dir
