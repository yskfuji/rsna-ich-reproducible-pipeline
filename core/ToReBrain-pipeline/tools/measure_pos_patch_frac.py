from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import typer
import yaml
from typer.models import OptionInfo

from src.datasets.isles_dataset import IslesVolumeDataset
from src.training.train_3d_unet import sample_patch_3d
from tools.meta_store import init_or_load_run
from tools.plotting import save_hist_png
from tools.run_meta import meta_to_dict, write_json, write_text

app = typer.Typer(add_completion=False)


def _coerce_option_default(v: Any) -> Any:
    return v.default if isinstance(v, OptionInfo) else v


def _parse_patch_size(s: str) -> Tuple[int, int, int]:
    xs = [int(x) for x in s.split(",") if x.strip()]
    if len(xs) != 3:
        raise ValueError(f"patch_size must have 3 ints (D,H,W), got: {s!r}")
    return xs[0], xs[1], xs[2]


def _resolve_paths(cfg: dict[str, Any], repo_root: Path) -> tuple[Path, Path]:
    csv_path = Path(cfg["data"]["csv_path"]).expanduser()
    root = Path(cfg["data"]["root"]).expanduser()
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    return csv_path, root


@app.command()
def main(
    config: str = typer.Option(..., help="Path to YAML config"),
    out_root: str = typer.Option("runs", help="Base output dir (runs/<run_id>/...)"),
    run_id: str = typer.Option("", help="Reuse an existing run_id (optional)"),
    seed: int = typer.Option(42, help="seed"),
    dataset_hash_mode: str = typer.Option("stat", help="dataset hash mode: stat|full"),
    splits: str = typer.Option("train,val,test", help="comma splits"),
    samples_per_split: int = typer.Option(2000, help="#patch samples per split"),
    # optional overrides
    patch_size: str = typer.Option("", help="override patch_size D,H,W"),
    pos_patch_frac: float = typer.Option(-1.0, help="override pos_patch_frac (0..1); -1 uses config"),
    bg_patch_size: str = typer.Option("", help="override bg_patch_size D,H,W (empty disables)"),
):
    # When called as a Python function (not via Typer CLI), defaults may be OptionInfo.
    splits = str(_coerce_option_default(splits))
    samples_per_split = int(_coerce_option_default(samples_per_split) or 0)
    patch_size = str(_coerce_option_default(patch_size))
    pos_patch_frac = float(_coerce_option_default(pos_patch_frac))
    bg_patch_size = str(_coerce_option_default(bg_patch_size))

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(config).expanduser().resolve() if Path(config).expanduser().is_absolute() else (repo_root / config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    csv_path, root = _resolve_paths(cfg, repo_root)

    ps = _parse_patch_size(patch_size) if patch_size.strip() else tuple(cfg["data"]["patch_size"])  # type: ignore[assignment]
    ppf = float(pos_patch_frac) if pos_patch_frac >= 0 else float(cfg["data"].get("pos_patch_frac", 0.5))
    bg_ps = _parse_patch_size(bg_patch_size) if bg_patch_size.strip() else tuple(cfg["data"].get("bg_patch_size", ps))

    fg_component_sampling = str(cfg.get("data", {}).get("fg_component_sampling", "uniform"))
    fg_component_sampling_alpha = float(cfg.get("data", {}).get("fg_component_sampling_alpha", 1.0))

    cfg_obj: dict[str, Any] = {
        "task": "measure_pos_patch_frac",
        "config": str(cfg_path),
        "csv_path": str(csv_path),
        "root": str(root),
        "splits": splits,
        "samples_per_split": int(samples_per_split),
        "patch_size": list(ps),
        "pos_patch_frac": float(ppf),
        "bg_patch_size": list(bg_ps),
        "fg_component_sampling": fg_component_sampling,
        "fg_component_sampling_alpha": float(fg_component_sampling_alpha),
    }

    meta, run_dir = init_or_load_run(
        repo_root=repo_root,
        out_root=Path(out_root),
        run_id=(run_id.strip() or None),
        seed=seed,
        config_path=cfg_path,
        config_obj=cfg_obj,
        csv_path=csv_path,
        data_root=root,
        dataset_hash_mode=dataset_hash_mode,
    )

    out_dir = run_dir / "sampling"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config_used.json", {"meta": meta_to_dict(meta), "config": cfg_obj})

    rng = np.random.default_rng(int(meta.seed))
    split_list = [s.strip() for s in splits.split(",") if s.strip()]

    results: dict[str, Any] = {"meta": meta_to_dict(meta), "task": "measure_pos_patch_frac", "per_split": {}}

    for split in split_list:
        ds = IslesVolumeDataset(
            str(csv_path),
            split=split,
            root=str(root),
            transform=None,
            normalize=str(cfg["data"].get("normalize", "legacy_zscore")),
            allow_missing_label=bool(cfg["data"].get("allow_missing_label", False)),
        )
        if len(ds) == 0:
            results["per_split"][split] = None
            continue

        pos_hits = 0
        fg_vox_list: list[int] = []
        fg_cc_size_list: list[int] = []
        for k in range(int(samples_per_split)):
            idx = int(rng.integers(0, len(ds)))
            item = ds[idx]
            img = item["image"].astype(np.float32)
            m = (item["mask"] > 0.5).astype(np.uint8)

            want_pos = bool(rng.random() < float(ppf))
            dbg: dict[str, Any] = {}
            patch_img, patch_mask = sample_patch_3d(
                img,
                m,
                patch_size=ps,
                foreground_prob=float(ppf),
                force_fg=want_pos,
                bg_patch_size=bg_ps,
                fg_component_sampling=fg_component_sampling,
                fg_component_sampling_alpha=float(fg_component_sampling_alpha),
                debug_meta=dbg,
            )
            fg_vox = int((patch_mask > 0.5).sum())
            fg_vox_list.append(fg_vox)
            if fg_vox > 0:
                pos_hits += 1
            if bool(want_pos) and ("fg_cc_size" in dbg):
                try:
                    fg_cc_size_list.append(int(dbg["fg_cc_size"]))
                except Exception:
                    pass

        est = float(pos_hits / max(1, int(samples_per_split)))
        results["per_split"][split] = {
            "samples": int(samples_per_split),
            "pos_hits": int(pos_hits),
            "estimated_pos_patch_frac": est,
            "fg_vox": {
                "mean": float(np.mean(fg_vox_list)) if fg_vox_list else None,
                "p50": float(np.percentile(fg_vox_list, 50)) if fg_vox_list else None,
                "p90": float(np.percentile(fg_vox_list, 90)) if fg_vox_list else None,
                "p99": float(np.percentile(fg_vox_list, 99)) if fg_vox_list else None,
                "max": int(np.max(fg_vox_list)) if fg_vox_list else None,
            },
            "fg_cc_size": {
                "sampling": fg_component_sampling,
                "alpha": float(fg_component_sampling_alpha),
                "n": int(len(fg_cc_size_list)),
                "p10": float(np.percentile(fg_cc_size_list, 10)) if fg_cc_size_list else None,
                "p50": float(np.percentile(fg_cc_size_list, 50)) if fg_cc_size_list else None,
                "p90": float(np.percentile(fg_cc_size_list, 90)) if fg_cc_size_list else None,
                "max": int(np.max(fg_cc_size_list)) if fg_cc_size_list else None,
            },
        }

        if fg_vox_list:
            save_hist_png(
                out_dir / f"{split}_fg_vox_hist.png",
                np.array(fg_vox_list, dtype=np.float32),
                title=f"{split}: fg voxels per sampled patch",
                xlabel="fg_vox",
                bins=60,
                logy=True,
            )

        if fg_cc_size_list:
            save_hist_png(
                out_dir / f"{split}_fg_cc_size_hist.png",
                np.array(fg_cc_size_list, dtype=np.float32),
                title=f"{split}: chosen FG CC size (force_fg only)",
                xlabel="fg_cc_size_vox",
                bins=60,
                logy=True,
            )

    write_json(out_dir / "summary.json", results)

    md_lines = [f"# Sampling Report ({meta.run_id})", ""]
    md_lines.append(f"- patch_size: `{list(ps)}`")
    md_lines.append(f"- pos_patch_frac (target): `{float(ppf)}`")
    md_lines.append(f"- bg_patch_size: `{list(bg_ps)}`")
    md_lines.append("")
    md_lines.append("## Per split")
    for split in split_list:
        r = results["per_split"].get(split)
        md_lines.append(f"- {split}: `{r}`")
    md_lines.append("")
    md_lines.append("## Outputs")
    md_lines.append(f"- `{out_dir / 'summary.json'}`")
    for split in split_list:
        md_lines.append(f"- `{out_dir / (split + '_fg_vox_hist.png')}`")
        md_lines.append(f"- `{out_dir / (split + '_fg_cc_size_hist.png')}`")
    write_text(out_dir / "report.md", "\n".join(md_lines) + "\n")


if __name__ == "__main__":
    app()
