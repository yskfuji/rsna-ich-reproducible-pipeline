from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml

from src.datasets.isles_dataset import IslesVolumeDataset
from tools.meta_store import init_or_load_run
from tools.plotting import save_hist_png
from tools.run_meta import meta_to_dict, sha256_json, write_json, write_text

app = typer.Typer(add_completion=False)


def _as_serializable(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return {
            "dtype": str(x.dtype),
            "shape": list(x.shape),
            "min": float(np.min(x)) if x.size else None,
            "max": float(np.max(x)) if x.size else None,
        }
    return x


def _preprocess_pipeline_id(cfg: dict) -> str:
    data = cfg.get("data", {})
    payload = {
        "modalities": list(data.get("modalities", [])),
        "normalize": data.get("normalize", "legacy_zscore"),
        "allow_missing_label": bool(data.get("allow_missing_label", False)),
        # If you later add resampling/cropping params to the dataset, include them here.
    }
    return sha256_json(payload)


@app.command()
def main(
    config: str = typer.Option(..., help="Path to YAML config"),
    out_root: str = typer.Option("runs", help="Base output dir (runs/<run_id>/...)"),
    run_id: str = typer.Option("", help="Reuse an existing run_id (optional)"),
    seed: int = typer.Option(42, help="seed to record"),
    dataset_hash_mode: str = typer.Option("stat", help="dataset hash mode: stat|full"),
):
    cfg_path = Path(config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    repo_root = Path(__file__).resolve().parents[1]

    data_root = Path(cfg["data"]["root"]).expanduser()
    csv_path = Path(cfg["data"]["csv_path"]).expanduser()
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()

    meta, run_dir = init_or_load_run(
        repo_root=repo_root,
        out_root=Path(out_root),
        run_id=(run_id.strip() or None),
        seed=int(seed),
        config_path=cfg_path,
        config_obj=cfg,
        csv_path=csv_path,
        data_root=data_root,
        dataset_hash_mode=dataset_hash_mode,
    )

    out_base = run_dir / "loader_check"
    out_base.mkdir(parents=True, exist_ok=True)

    pipeline_id = _preprocess_pipeline_id(cfg)

    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        ds = IslesVolumeDataset(
            str(csv_path),
            split=split,
            root=str(data_root),
            transform=None,
            normalize=cfg.get("data", {}).get("normalize", "legacy_zscore"),
            allow_missing_label=bool(cfg.get("data", {}).get("allow_missing_label", False)),
        )
        sample = ds[0]
        img = sample["image"]
        mask = sample["mask"]

        mask_unique = np.unique(mask.astype(np.float32))
        rows.append(
            {
                **meta_to_dict(meta),
                "split": split,
                "case_id": sample.get("case_id"),
                "preprocess_pipeline_id": pipeline_id,
                "image": {
                    "dtype": str(img.dtype),
                    "shape": list(img.shape),
                    "min": float(np.min(img)),
                    "max": float(np.max(img)),
                },
                "mask": {
                    "dtype": str(mask.dtype),
                    "shape": list(mask.shape),
                    "min": float(np.min(mask)),
                    "max": float(np.max(mask)),
                    "unique": mask_unique[:50].tolist(),
                },
            }
        )

        # quick visualization: histogram of first modality
        save_hist_png(
            out_base / f"loader_check_hist_{split}.png",
            img[0],
            title=f"{split}: image[0] histogram",
            xlabel="value",
            bins=80,
            logy=True,
        )

    # assert modality order if config declares it
    mods = cfg.get("data", {}).get("modalities", [])
    if mods:
        want = ["DWI", "ADC", "FLAIR"]
        if [str(m).upper() for m in mods] != want:
            raise ValueError(f"Modalities order must be {want}, got: {mods}")

    # ensure preprocess pipeline id identical
    ids = {r["preprocess_pipeline_id"] for r in rows}
    if len(ids) != 1:
        raise AssertionError(f"preprocess_pipeline_id differs across splits: {ids}")

    out_json = out_base / "loader_check.json"
    write_json(out_json, {"meta": meta_to_dict(meta), "checks": rows})

    md = "\n".join(
        [
            f"# Loader Check ({meta.run_id})",
            "",
            f"- git_commit: `{meta.git_commit}`",
            f"- config_hash: `{meta.config_hash}`",
            f"- dataset_hash: `{meta.dataset_hash}` ({meta.dataset_hash_mode})",
            f"- preprocess_pipeline_id: `{pipeline_id}`",
            "",
            "## First-sample summary",
        ]
        + [
            f"- {r['split']}: case_id={r['case_id']} image_shape={r['image']['shape']} mask_shape={r['mask']['shape']} mask_unique_head={r['mask']['unique'][:5]}"
            for r in rows
        ]
        + ["", f"JSON: `{out_json}`"]
    )
    write_text(out_base / "loader_check.md", md + "\n")


if __name__ == "__main__":
    app()
