from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import typer
import yaml
from typer.models import OptionInfo
from numpy.typing import NDArray
from typing import cast, Literal

from src.preprocess.utils_io import load_nifti
from tools.meta_store import init_or_load_run
from tools.plotting import save_hist_png
from tools.run_meta import meta_to_dict, write_json, write_text

app = typer.Typer(add_completion=False)


def _coerce_option_default(v: Any) -> Any:
    return v.default if isinstance(v, OptionInfo) else v


def _axis_codes(affine: NDArray[np.floating]) -> tuple[str, str, str]:
    import nibabel as nib

    return tuple(str(x) for x in nib.aff2axcodes(affine))  # type: ignore[return-value]


def _voxel_sizes(img: Any) -> tuple[float, float, float]:
    import nibabel as nib

    zooms = img.header.get_zooms()[:3]
    return float(zooms[0]), float(zooms[1]), float(zooms[2])


def _percentiles(x: NDArray[np.floating], ps: Iterable[float]) -> dict[str, float]:
    vals = np.percentile(x, list(ps)).astype(np.float64)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def _median_iqr(x: NDArray[np.floating]) -> dict[str, float | None]:
    x = x.astype(np.float64, copy=False).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": None, "iqr": None, "q25": None, "q75": None}
    q25, med, q75 = np.percentile(x, [25, 50, 75]).astype(np.float64)
    return {"median": float(med), "iqr": float(q75 - q25), "q25": float(q25), "q75": float(q75)}


def _stats(x: NDArray[np.floating]) -> dict[str, Any]:
    x = x.astype(np.float32, copy=False)
    if x.size == 0:
        return {"mean": None, "std": None, "p1": None, "p5": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        **_percentiles(x, [1, 5, 50, 95, 99]),
    }


def _sat_fracs(x: NDArray[np.floating]) -> dict[str, float | None]:
    x = x.astype(np.float32, copy=False)
    if x.size == 0:
        return {"sat0_frac": None, "sat1_frac": None}
    v0 = cast(np.floating, np.mean((x == 0).astype(np.float32)))
    v1 = cast(np.floating, np.mean((x == 1).astype(np.float32)))
    sat0 = float(v0)
    sat1 = float(v1)
    return {"sat0_frac": sat0, "sat1_frac": sat1}


def _resolve_paths(cfg: dict[str, Any], repo_root: Path) -> tuple[Path, Path, list[str]]:
    data_root = Path(cfg["data"]["root"]).expanduser()
    csv_path = Path(cfg["data"]["csv_path"]).expanduser()
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()
    mods = [str(m) for m in cfg.get("data", {}).get("modalities", [])]
    return data_root, csv_path, mods


@app.command()
def main(
    config: str = typer.Option(..., help="Path to YAML config"),
    out_root: str = typer.Option("runs", help="Base output dir (runs/<run_id>/...)"),
    run_id: str = typer.Option("", help="Reuse an existing run_id (optional)"),
    seed: int = typer.Option(42, help="seed"),
    dataset_hash_mode: str = typer.Option("stat", help="dataset hash mode: stat|full"),
    max_cases_per_split: int = typer.Option(0, help="limit cases per split (0 = all)"),
    sample_voxels_per_split: int = typer.Option(200_000, help="voxels to sample for histograms per split per channel"),
):
    # When called as a Python function (not via Typer CLI), defaults may be OptionInfo.
    max_cases_per_split = int(_coerce_option_default(max_cases_per_split) or 0)
    sample_voxels_per_split = int(_coerce_option_default(sample_voxels_per_split) or 0)

    cfg_path = Path(config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    dataset_hash_mode_str = str(_coerce_option_default(dataset_hash_mode) or "stat").strip()
    if dataset_hash_mode_str not in {"stat", "full"}:
        raise ValueError(f"dataset_hash_mode must be 'stat' or 'full', got: {dataset_hash_mode_str}")
    dataset_hash_mode_lit = cast(Literal["stat", "full"], dataset_hash_mode_str)

    repo_root = Path(__file__).resolve().parents[1]
    data_root, csv_path, mods = _resolve_paths(cfg, repo_root)

    run_id_opt = run_id.strip() or None
    meta, run_dir = init_or_load_run(
        repo_root=repo_root,
        out_root=Path(out_root),
        run_id=run_id_opt,
        seed=seed,
        config_path=cfg_path,
        config_obj=cfg,
        csv_path=csv_path,
        data_root=data_root,
        dataset_hash_mode=dataset_hash_mode_lit,
    )

    out_dir = run_dir / "data_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_csv(str(csv_path))

    case_stats_path = out_dir / "case_stats.jsonl"
    with case_stats_path.open("w", encoding="utf-8") as f:
        split_summaries: dict[str, Any] = {}
        for split in ["train", "val", "test"]:
            df_s = df[df["split"] == split].reset_index(drop=True)
            if max_cases_per_split and len(df_s) > max_cases_per_split:
                df_s = df_s.iloc[: int(max_cases_per_split)].reset_index(drop=True)

            sampled_per_ch: list[list[float]] = []
            sat0_per_ch: list[list[float]] = []
            sat1_per_ch: list[list[float]] = []
            voxsz_list: list[tuple[float, float, float]] = []
            axcodes_list: list[tuple[str, str, str]] = []
            spatial_shape_list: list[tuple[int, int, int]] = []
            # determine channel count from first case
            if len(df_s) == 0:
                continue
            first_case = str(df_s.iloc[0]["case_id"])
            first_img_path = data_root / "images" / f"{first_case}.nii.gz"
            first_arr, _ = load_nifti(str(first_img_path))
            ch = int(first_arr.shape[0]) if first_arr.ndim == 4 else 1
            sampled_per_ch = [[] for _ in range(ch)]
            sat0_per_ch = [[] for _ in range(ch)]
            sat1_per_ch = [[] for _ in range(ch)]

            per_case_rows: list[dict[str, Any]] = []
            for _, row in df_s.iterrows():
                case_id = str(row["case_id"])
                img_path = data_root / "images" / f"{case_id}.nii.gz"
                lbl_path = data_root / "labels" / f"{case_id}.nii.gz"

                img_arr, img = load_nifti(str(img_path))
                lbl_arr, _ = load_nifti(str(lbl_path)) if lbl_path.exists() else (None, None)

                if img_arr.ndim == 3:
                    img_arr = img_arr[None, ...]

                shape = list(img_arr.shape)
                vox = _voxel_sizes(img)
                affine = np.asarray(img.affine if getattr(img, "affine", None) is not None else np.eye(4), dtype=np.float64)
                ax = _axis_codes(affine)
                det = float(np.linalg.det(affine[:3, :3]))

                voxsz_list.append(vox)
                axcodes_list.append(ax)
                spatial_shape_list.append((int(img_arr.shape[-3]), int(img_arr.shape[-2]), int(img_arr.shape[-1])))

                per_mod: list[dict[str, Any]] = []
                for c in range(img_arr.shape[0]):
                    vol = img_arr[c].astype(np.float32, copy=False)
                    nz = vol[vol != 0]
                    sat = _sat_fracs(vol)
                    per_mod.append(
                        {
                            "channel": int(c),
                            "modality": mods[c] if c < len(mods) else None,
                            "all": _stats(vol),
                            "nonzero": _stats(nz) if nz.size else _stats(nz),
                            **sat,
                        }
                    )

                    if sat["sat0_frac"] is not None:
                        sat0_per_ch[c].append(float(sat["sat0_frac"]))
                    if sat["sat1_frac"] is not None:
                        sat1_per_ch[c].append(float(sat["sat1_frac"]))

                    # histogram sample
                    if sample_voxels_per_split > 0:
                        rng = np.random.default_rng(int(meta.seed) + 17 * c)
                        vals = nz if nz.size else vol.ravel()
                        if vals.size:
                            k = int(min(sample_voxels_per_split // max(1, len(df_s)), vals.size, 5000))
                            if k > 0:
                                idxs = rng.integers(0, vals.size, size=k)
                                sampled_per_ch[c].extend(vals[idxs].astype(np.float32).tolist())

                gt_vox = int(np.sum(lbl_arr > 0.5)) if lbl_arr is not None else None

                rec = {
                    **meta_to_dict(meta),
                    "task": "data_report",
                    "split": split,
                    "case_id": case_id,
                    "image_path": str(img_path),
                    "label_path": str(lbl_path) if lbl_path.exists() else None,
                    "shape": shape,
                    "voxel_sizes": list(vox),
                    "axis_codes": list(ax),
                    "affine_det": det,
                    "per_modality": per_mod,
                    "gt_vox": gt_vox,
                }
                per_case_rows.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # split summary
            gt_vox_list = [r["gt_vox"] for r in per_case_rows if r.get("gt_vox") is not None]

            voxsz_arr = np.array(voxsz_list, dtype=np.float64) if voxsz_list else np.zeros((0, 3), dtype=np.float64)
            shp_arr = np.array(spatial_shape_list, dtype=np.float64) if spatial_shape_list else np.zeros((0, 3), dtype=np.float64)
            ax_counts: dict[str, int] = {}
            for a in axcodes_list:
                key = "".join(a)
                ax_counts[key] = ax_counts.get(key, 0) + 1

            saturation_by_ch: list[dict[str, Any]] = []
            for c in range(ch):
                s0 = np.array(sat0_per_ch[c], dtype=np.float64)
                s1 = np.array(sat1_per_ch[c], dtype=np.float64)
                saturation_by_ch.append(
                    {
                        "channel": int(c),
                        "modality": mods[c] if c < len(mods) else None,
                        "sat0": {
                            "mean": float(s0.mean()) if s0.size else None,
                            **_median_iqr(s0),
                        },
                        "sat1": {
                            "mean": float(s1.mean()) if s1.size else None,
                            **_median_iqr(s1),
                        },
                    }
                )

            split_summaries[split] = {
                "n_cases": int(len(per_case_rows)),
                "voxel_sizes_mm": {
                    "x": {"mean": float(voxsz_arr[:, 0].mean()) if voxsz_arr.size else None, **_median_iqr(voxsz_arr[:, 0] if voxsz_arr.size else np.array([], dtype=np.float64))},
                    "y": {"mean": float(voxsz_arr[:, 1].mean()) if voxsz_arr.size else None, **_median_iqr(voxsz_arr[:, 1] if voxsz_arr.size else np.array([], dtype=np.float64))},
                    "z": {"mean": float(voxsz_arr[:, 2].mean()) if voxsz_arr.size else None, **_median_iqr(voxsz_arr[:, 2] if voxsz_arr.size else np.array([], dtype=np.float64))},
                },
                "spatial_shape": {
                    "d": {"mean": float(shp_arr[:, 0].mean()) if shp_arr.size else None, **_median_iqr(shp_arr[:, 0] if shp_arr.size else np.array([], dtype=np.float64))},
                    "h": {"mean": float(shp_arr[:, 1].mean()) if shp_arr.size else None, **_median_iqr(shp_arr[:, 1] if shp_arr.size else np.array([], dtype=np.float64))},
                    "w": {"mean": float(shp_arr[:, 2].mean()) if shp_arr.size else None, **_median_iqr(shp_arr[:, 2] if shp_arr.size else np.array([], dtype=np.float64))},
                },
                "axis_codes": {
                    "counts": dict(sorted(ax_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                },
                "saturation": {
                    "by_channel": saturation_by_ch,
                },
                "gt_vox": {
                    "min": int(np.min(gt_vox_list)) if gt_vox_list else None,
                    "median": float(np.median(gt_vox_list)) if gt_vox_list else None,
                    "max": int(np.max(gt_vox_list)) if gt_vox_list else None,
                },
            }

            # save histograms per channel
            for c in range(ch):
                vals = np.array(sampled_per_ch[c], dtype=np.float32)
                title = f"{split}: {mods[c] if c < len(mods) else f'ch{c}'} nonzero histogram"
                save_hist_png(out_dir / f"hist_{split}_ch{c}.png", vals, title=title, xlabel="value", bins=80, logy=True)

        # write split summary
        write_json(out_dir / "split_summary.json", {"meta": meta_to_dict(meta), "splits": split_summaries})

    # markdown report
    md_lines = [
        f"# Data Report ({meta.run_id})",
        "",
        f"- git_commit: `{meta.git_commit}`",
        f"- config_hash: `{meta.config_hash}`",
        f"- dataset_hash: `{meta.dataset_hash}` ({meta.dataset_hash_mode})",
        "",
        "## Outputs",
        f"- `{case_stats_path}`",
        f"- `{out_dir / 'split_summary.json'}`",
    ]
    for split in ["train", "val", "test"]:
        for c in range(8):
            p = out_dir / f"hist_{split}_ch{c}.png"
            if p.exists():
                md_lines.append(f"- `{p}`")
    write_text(out_dir / "report.md", "\n".join(md_lines) + "\n")


if __name__ == "__main__":
    app()
