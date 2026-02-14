from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple, Literal, cast

import numpy as np
import typer
from typer.models import OptionInfo

from tools.inference_log import (
    _filter_cc_score,
    _filter_min_size,
    _infer_logits_dispatch,
    _load_temperature,
    _parse_patch_size,
)
from tools.meta_store import init_or_load_run
from tools.plotting import save_curve_png
from tools.run_meta import meta_to_dict, write_json, write_text

from src.datasets.isles_dataset import IslesVolumeDataset
from src.models.unet_3d import UNet3D
from src.training.utils_train import prepare_device

import torch

app = typer.Typer(add_completion=False)


def _coerce_option_default(v: Any) -> Any:
    return v.default if isinstance(v, OptionInfo) else v


def _threshold_grid(step: float) -> list[float]:
    step_f = float(step)
    if not np.isfinite(step_f) or step_f <= 0:
        raise ValueError(f"step must be positive, got: {step}")
    n = int(round(1.0 / step_f))
    xs = [round(i * step_f, 10) for i in range(1, n)]
    xs = [float(x) for x in xs if 0.0 < x < 1.0]
    if 0.5 not in xs:
        xs.append(0.5)
    return sorted(set(xs))


def _load_model(mp: Path, in_ch: int, device: torch.device) -> torch.nn.Module:
    state = torch.load(str(mp), map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    w = state_dict.get("enc1.0.weight")
    if w is None:
        raise RuntimeError("Cannot infer base_ch from checkpoint; missing enc1.0.weight")
    base_ch = int(w.shape[0])
    model = UNet3D(in_channels=in_ch, out_channels=1, base_ch=base_ch, deep_supervision=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@app.command()
def main(
    model_path: str = typer.Option(..., help="model checkpoint"),
    csv_path: str = typer.Option(..., help="split csv"),
    root: str = typer.Option(..., help="processed root"),
    split: str = typer.Option("val", help="train|val|test"),
    out_root: str = typer.Option("runs", help="Base output dir (runs/<run_id>/...)"),
    run_id: str = typer.Option("", help="Reuse an existing run_id (optional)"),
    out_subdir: str = typer.Option("", help="output subdir under runs/<run_id>/ (default: threshold_sweep)"),
    seed: int = typer.Option(42, help="seed"),
    dataset_hash_mode: str = typer.Option("stat", help="dataset hash mode: stat|full"),
    patch_size: str = typer.Option("64,64,48", help="patch size"),
    overlap: float = typer.Option(0.5, help="overlap"),
    step: float = typer.Option(0.05, help="threshold step (0.05 => 0.05..0.95)"),
    min_size: int = typer.Option(0, help="min component size"),
    cc_score: str = typer.Option("none", help="none|max_prob|p95_prob|mean_prob"),
    cc_score_thr: float = typer.Option(0.5, help="component score threshold"),
    temperature: str = typer.Option("1.0", help="float or from_run_best/from_run_last"),
    normalize: str = typer.Option("nonzero_zscore", help="legacy_zscore|nonzero_zscore|none"),
    allow_missing_label: bool = typer.Option(False, help="missing label -> all zero"),
    tta: str = typer.Option("full", help="full|flip|none"),
    quiet: bool = typer.Option(False, help="suppress per-case prints"),
):
    # When called as a Python function (not via Typer CLI), defaults may be OptionInfo.
    step = float(_coerce_option_default(step))
    min_size = int(_coerce_option_default(min_size) or 0)
    cc_score = str(_coerce_option_default(cc_score))
    cc_score_thr = float(_coerce_option_default(cc_score_thr))
    temperature = str(_coerce_option_default(temperature))
    allow_missing_label = bool(_coerce_option_default(allow_missing_label))
    quiet = bool(_coerce_option_default(quiet))
    out_subdir = str(_coerce_option_default(out_subdir) or "").strip()

    dataset_hash_mode_str = str(_coerce_option_default(dataset_hash_mode) or "stat").strip()
    if dataset_hash_mode_str not in {"stat", "full"}:
        raise ValueError(f"dataset_hash_mode must be 'stat' or 'full', got: {dataset_hash_mode_str}")
    dataset_hash_mode_lit = cast(Literal["stat", "full"], dataset_hash_mode_str)

    repo_root = Path(__file__).resolve().parents[1]
    mp = Path(model_path).expanduser().resolve()
    cp = Path(csv_path).expanduser().resolve() if Path(csv_path).expanduser().is_absolute() else (repo_root / csv_path).resolve()
    data_root = Path(root).expanduser().resolve() if Path(root).expanduser().is_absolute() else (repo_root / root).resolve()

    run_id_opt = run_id.strip() or None
    cfg_obj: dict[str, Any] = {
        "task": "threshold_sweep",
        "model_path": str(mp),
        "csv_path": str(cp),
        "root": str(data_root),
        "split": split,
        "patch_size": patch_size,
        "overlap": float(overlap),
        "step": float(step),
        "min_size": int(min_size),
        "cc_score": cc_score,
        "cc_score_thr": float(cc_score_thr),
        "temperature": temperature,
        "normalize": normalize,
        "allow_missing_label": bool(allow_missing_label),
        "tta": tta,
    }

    meta, run_dir = init_or_load_run(
        repo_root=repo_root,
        out_root=Path(out_root),
        run_id=run_id_opt,
        seed=seed,
        config_path=None,
        config_obj=cfg_obj,
        csv_path=cp,
        data_root=data_root,
        dataset_hash_mode=dataset_hash_mode_lit,
    )

    out_dir = run_dir / (out_subdir or "threshold_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config_used.json", {"meta": meta_to_dict(meta), "config": cfg_obj})

    ps: Tuple[int, int, int] = _parse_patch_size(patch_size)
    thr_list = _threshold_grid(step)
    temp = _load_temperature(mp, temperature)

    device = prepare_device()
    ds = IslesVolumeDataset(
        str(cp),
        split=split,
        root=str(data_root),
        transform=None,
        normalize=normalize,
        allow_missing_label=bool(allow_missing_label),
    )
    first = ds[0]["image"]
    in_ch = int(first.shape[0]) if first.ndim == 4 else 1
    model = _load_model(mp, in_ch=in_ch, device=device)

    # accumulators
    agg = {
        float(thr): {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "gt_pos": 0,
            "gt_neg": 0,
            "det_pos": 0,
            "det_neg": 0,
            "sum_dice": 0.0,
            "n": 0,
        }
        for thr in thr_list
    }

    for i in range(len(ds)):
        item = ds[i]
        vol = item["image"].astype(np.float32)
        gt = (item["mask"] > 0.5).astype(np.uint8)
        if vol.ndim == 3:
            vol = vol[None, ...]

        logits = _infer_logits_dispatch(vol, model, patch_size=ps, overlap=float(overlap), device=device, tta=tta)
        logits = logits[0, 0].astype(np.float32)
        prob = (1.0 / (1.0 + np.exp(-(logits / float(temp))))).astype(np.float32)

        gt_pos = bool(int(gt.sum()) > 0)

        for thr in thr_list:
            thr_f = float(thr)
            pred = (prob >= thr_f).astype(np.uint8)
            pred = _filter_min_size(pred, int(min_size))
            pred = _filter_cc_score(pred, prob, cc_score, float(cc_score_thr))

            tp = int(((pred > 0) & (gt > 0)).sum())
            fp = int(((pred > 0) & (gt == 0)).sum())
            fn = int(((pred == 0) & (gt > 0)).sum())
            dice = float((2 * tp) / (2 * tp + fp + fn + 1e-8))

            a = agg[thr_f]
            a["tp"] += tp
            a["fp"] += fp
            a["fn"] += fn
            a["gt_pos"] += int(gt_pos)
            a["gt_neg"] += int(not gt_pos)
            a["det_pos"] += int(gt_pos and int(pred.sum()) > 0)
            a["det_neg"] += int((not gt_pos) and int(pred.sum()) > 0)
            a["sum_dice"] += dice
            a["n"] += 1

        if not quiet:
            print(f"[thr_sweep] {split} {i+1}/{len(ds)} {item['case_id']}", flush=True)

    per_threshold: list[dict[str, Any]] = []
    for thr in thr_list:
        a = agg[float(thr)]
        n = max(1, int(a["n"]))
        tp, fp, fn = int(a["tp"]), int(a["fp"]), int(a["fn"])
        prec = float(tp / (tp + fp + 1e-8))
        rec = float(tp / (tp + fn + 1e-8))
        det_rate_case = float(a["det_pos"] / max(1, int(a["gt_pos"]))) if a["gt_pos"] else None
        far_case = float(a["det_neg"] / max(1, int(a["gt_neg"]))) if a["gt_neg"] else None
        per_threshold.append(
            {
                "threshold": float(thr),
                "mean_dice": float(a["sum_dice"] / n),
                "voxel_precision": prec,
                "voxel_recall": rec,
                "detection_rate_case": det_rate_case,
                "false_alarm_rate_case": far_case,
                "n_cases": int(a["n"]),
            }
        )

    per_threshold_sorted = sorted(per_threshold, key=lambda r: (r["mean_dice"], -r["threshold"]), reverse=True)
    best = per_threshold_sorted[0] if per_threshold_sorted else None

    write_json(
        out_dir / "curve.json",
        {
            "meta": meta_to_dict(meta),
            "split": split,
            "per_threshold": per_threshold,
            "best": best,
            "settings": {
                "threshold_step": float(step),
                "min_size": int(min_size),
                "cc_score": cc_score,
                "cc_score_thr": float(cc_score_thr),
                "temperature": float(temp),
                "patch_size": list(ps),
                "overlap": float(overlap),
                "tta": tta,
            },
        },
    )

    if per_threshold:
        xs = [float(r["threshold"]) for r in sorted(per_threshold, key=lambda r: r["threshold"])]
        ys = [float(r["mean_dice"]) for r in sorted(per_threshold, key=lambda r: r["threshold"])]
        save_curve_png(
            out_dir / "curve_mean_dice.png",
            xs=xs,
            ys=ys,
            title=f"{split}: mean Dice vs threshold",
            xlabel="threshold",
            ylabel="mean Dice",
        )

    write_json(out_dir / "best_threshold.json", {"meta": meta_to_dict(meta), "best": best})

    md = "\n".join(
        [
            f"# Threshold Sweep ({meta.run_id})",
            "",
            f"- split: `{split}`",
            f"- step: `{step}`",
            f"- best: `{best}`" if best is not None else "- best: `None`",
            "",
            "## Outputs",
            f"- `{out_dir / 'curve.json'}`",
            f"- `{out_dir / 'curve_mean_dice.png'}`",
            f"- `{out_dir / 'best_threshold.json'}`",
        ]
    )
    write_text(out_dir / "report.md", md + "\n")


if __name__ == "__main__":
    app()
