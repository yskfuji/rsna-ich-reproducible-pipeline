from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Tuple, cast

import numpy as np
from numpy.typing import NDArray
import torch
import typer
from typer.models import OptionInfo

from src.datasets.isles_dataset import IslesVolumeDataset
from src.evaluation.evaluate_isles import (
    infer_logits,
    infer_logits_with_flip_tta,
    infer_logits_with_tta,
)
from src.evaluation.metrics_segmentation import dice_score
from src.models.unet_3d import UNet3D
from src.preprocess.utils_io import load_nifti, save_nifti
from src.training.utils_train import prepare_device
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


def _parse_thresholds(model_path: Path, thresholds: str) -> list[float]:
    t = thresholds.strip().lower()
    if t in {"from_run_best", "from_run_last"}:
        run_dir = model_path.parent
        meta_name = "val_threshold_best.json" if t == "from_run_best" else "val_threshold_last.json"
        meta_path = run_dir / meta_name
        meta = json.loads(meta_path.read_text())
        return [float(meta["val_threshold"])]
    return [float(x) for x in thresholds.split(",") if x.strip()]


def _load_temperature(model_path: Path, temperature: str) -> float:
    t = str(temperature).strip().lower()
    if t in {"from_run_best", "from_run_last"}:
        run_dir = model_path.parent
        meta_name = "temperature_best.json" if t == "from_run_best" else "temperature_last.json"
        meta_path = run_dir / meta_name
        meta = json.loads(meta_path.read_text())
        return float(meta.get("temperature", 1.0))
    return float(temperature)


def _infer_logits_dispatch(
    vol: NDArray[np.float32],
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    tta: str,
) -> NDArray[np.float32]:
    t = tta.strip().lower()
    if t == "full":
        return infer_logits_with_tta(vol, model, patch_size=patch_size, overlap=overlap, device=device)
    if t == "flip":
        return infer_logits_with_flip_tta(vol, model, patch_size=patch_size, overlap=overlap, device=device)
    if t == "none":
        return infer_logits(vol, model, patch_size=patch_size, overlap=overlap, device=device)
    raise ValueError(f"Unknown tta: {tta!r}")


def _filter_min_size(pred: NDArray[np.uint8], min_size: int) -> NDArray[np.uint8]:
    from scipy.ndimage import label as cc_label  # type: ignore[import-not-found]

    if min_size <= 0:
        return pred
    res = cc_label(pred.astype(np.uint8))
    lbl = cast(NDArray[np.int64], res[0]).astype(np.int64, copy=False)  # type: ignore[index]
    if lbl.max() == 0:
        return pred
    sizes = np.bincount(lbl.ravel())
    remove = sizes < int(min_size)
    remove[0] = False
    out = pred.copy()
    out[remove[lbl]] = 0
    return out


def _filter_cc_score(
    pred: NDArray[np.uint8],
    probs: NDArray[np.float32],
    score_mode: str,
    score_thr: float,
) -> NDArray[np.uint8]:
    from scipy.ndimage import label as cc_label  # type: ignore[import-not-found]

    mode = score_mode.strip().lower()
    if mode in {"none", "off", "false"}:
        return pred
    res = cc_label(pred.astype(np.uint8))
    lbl = cast(NDArray[np.int64], res[0]).astype(np.int64, copy=False)  # type: ignore[index]
    if lbl.max() == 0:
        return pred

    out = pred.copy()
    thr = float(score_thr)
    for comp_id in range(1, int(lbl.max()) + 1):
        m = lbl == comp_id
        if not np.any(m):
            continue
        vals = probs[m]
        if vals.size == 0:
            out[m] = 0
            continue
        if mode == "max_prob":
            score = float(vals.max())
        elif mode == "mean_prob":
            score = float(vals.mean())
        elif mode == "p95_prob":
            score = float(np.percentile(vals, 95.0))
        else:
            raise ValueError(f"Unknown cc_score: {score_mode!r}")
        if score < thr:
            out[m] = 0
    return out


def _fp_component_stats(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> tuple[int, int]:
    from scipy.ndimage import label as cc_label  # type: ignore[import-not-found]

    fp = (pred > 0) & (gt == 0)
    if not np.any(fp):
        return 0, 0
    res = cc_label(fp.astype(np.uint8))
    lbl = cast(NDArray[np.int64], res[0]).astype(np.int64, copy=False)  # type: ignore[index]
    n = int(res[1])  # type: ignore[index]
    fp_vox = int(fp.sum())
    fp_cc = int(n)
    return fp_cc, fp_vox


@app.command()
def main(
    model_path: str = typer.Option(..., help="model checkpoint"),
    csv_path: str = typer.Option(..., help="split csv"),
    root: str = typer.Option(..., help="processed root"),
    split: str = typer.Option("test", help="train|val|test"),
    out_root: str = typer.Option("runs", help="Base output dir (runs/<run_id>/...)"),
    run_id: str = typer.Option("", help="Reuse an existing run_id (optional)"),
    seed: int = typer.Option(42, help="seed"),
    dataset_hash_mode: str = typer.Option("stat", help="dataset hash mode: stat|full"),
    patch_size: str = typer.Option("64,64,48", help="patch size"),
    overlap: float = typer.Option(0.5, help="overlap"),
    thresholds: str = typer.Option("0.5", help="comma thresholds or from_run_best/from_run_last"),
    min_size: int = typer.Option(0, help="min component size (0 disables)"),
    cc_score: str = typer.Option("none", help="none|max_prob|p95_prob|mean_prob"),
    cc_score_thr: float = typer.Option(0.5, help="component score threshold"),
    temperature: str = typer.Option("1.0", help="float or from_run_best/from_run_last"),
    normalize: str = typer.Option("nonzero_zscore", help="legacy_zscore|nonzero_zscore|none"),
    allow_missing_label: bool = typer.Option(False, help="missing label -> all zero"),
    tta: str = typer.Option("full", help="full|flip|none"),
    save_prob_maps: bool = typer.Option(False, help="save per-case prob NIfTI under inference/prob_maps"),
    quiet: bool = typer.Option(False, help="suppress per-case prints"),
):
    # When called as a Python function (not via Typer CLI), defaults may be OptionInfo.
    thresholds = str(_coerce_option_default(thresholds))
    min_size = int(_coerce_option_default(min_size) or 0)
    cc_score = str(_coerce_option_default(cc_score))
    cc_score_thr = float(_coerce_option_default(cc_score_thr))
    temperature = str(_coerce_option_default(temperature))
    allow_missing_label = bool(_coerce_option_default(allow_missing_label))
    save_prob_maps = bool(_coerce_option_default(save_prob_maps))
    quiet = bool(_coerce_option_default(quiet))

    repo_root = Path(__file__).resolve().parents[1]
    mp = Path(model_path).expanduser().resolve()
    cp = Path(csv_path).expanduser().resolve() if Path(csv_path).expanduser().is_absolute() else (repo_root / csv_path).resolve()
    data_root = Path(root).expanduser().resolve() if Path(root).expanduser().is_absolute() else (repo_root / root).resolve()

    run_id_opt = run_id.strip() or None
    # No YAML config necessarily; hash the CLI args as a config surrogate.
    cfg_obj = {
        "model_path": str(mp),
        "csv_path": str(cp),
        "root": str(data_root),
        "split": split,
        "patch_size": patch_size,
        "overlap": float(overlap),
        "thresholds": thresholds,
        "min_size": int(min_size),
        "cc_score": cc_score,
        "cc_score_thr": float(cc_score_thr),
        "temperature": temperature,
        "normalize": normalize,
        "allow_missing_label": bool(allow_missing_label),
        "tta": tta,
        "save_prob_maps": bool(save_prob_maps),
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
        dataset_hash_mode=dataset_hash_mode,
    )

    out_dir = run_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config_used.json", {"meta": meta_to_dict(meta), "config": cfg_obj})

    ps = _parse_patch_size(patch_size)
    thr_list = _parse_thresholds(mp, thresholds)
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

    state = torch.load(str(mp), map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    # infer base_ch
    w = state_dict.get("enc1.0.weight")
    if w is None:
        raise RuntimeError("Cannot infer base_ch from checkpoint; missing enc1.0.weight")
    base_ch = int(w.shape[0])
    deep_supervision = False
    model = UNet3D(in_channels=in_ch, out_channels=1, base_ch=base_ch, deep_supervision=deep_supervision)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # per-threshold aggregates
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
            "sum_prec": 0.0,
            "sum_rec": 0.0,
            "sum_fp_cc": 0,
            "sum_fp_vox": 0,
            "n": 0,
        }
        for thr in thr_list
    }

    per_case_path = out_dir / "per_case_metrics.jsonl"
    prob_maps_dir = out_dir / "prob_maps"
    if save_prob_maps:
        prob_maps_dir.mkdir(parents=True, exist_ok=True)

    prob_samples: list[float] = []

    with per_case_path.open("w", encoding="utf-8") as f:
        for i in range(len(ds)):
            item = ds[i]
            case_id = str(item["case_id"])
            vol = item["image"].astype(np.float32)
            gt = (item["mask"] > 0.5).astype(np.uint8)

            if vol.ndim == 3:
                vol = vol[None, ...]

            logits = _infer_logits_dispatch(vol, model, patch_size=ps, overlap=float(overlap), device=device, tta=tta)
            logits = logits[0, 0].astype(np.float32)

            # temperature scaling
            logits_t = logits / float(temp)
            prob = (1.0 / (1.0 + np.exp(-logits_t))).astype(np.float32)

            # collect prob sample for histogram
            if prob.size:
                rng = np.random.default_rng(int(meta.seed) + i)
                k = int(min(2000, prob.size))
                idxs = rng.integers(0, prob.size, size=k)
                prob_samples.extend(prob.ravel()[idxs].astype(np.float32).tolist())

            if save_prob_maps:
                img_path = data_root / "images" / f"{case_id}.nii.gz"
                _, ref = load_nifti(str(img_path))
                save_nifti(prob, ref, str(prob_maps_dir / f"{case_id}_prob.nii.gz"))

            gt_vox = int(gt.sum())
            gt_pos = gt_vox > 0

            # logits/prob summary
            logits_stats = {
                "min": float(np.min(logits)),
                "max": float(np.max(logits)),
                "mean": float(np.mean(logits)),
                "std": float(np.std(logits)),
            }
            prob_stats = {
                "min": float(np.min(prob)),
                "max": float(np.max(prob)),
                "mean": float(np.mean(prob)),
                "std": float(np.std(prob)),
            }

            for thr in thr_list:
                thr_f = float(thr)
                pred = (prob >= thr_f).astype(np.uint8)
                pred = _filter_min_size(pred, int(min_size))
                pred = _filter_cc_score(pred, prob, cc_score, float(cc_score_thr))

                pred_vox = int(pred.sum())
                tp = int(((pred > 0) & (gt > 0)).sum())
                fp = int(((pred > 0) & (gt == 0)).sum())
                fn = int(((pred == 0) & (gt > 0)).sum())

                dice = float(dice_score(pred, gt))
                prec = float(tp / (tp + fp + 1e-8))
                rec = float(tp / (tp + fn + 1e-8))

                fp_cc, fp_cc_vox = _fp_component_stats(pred, gt)

                a = agg[thr_f]
                a["tp"] += tp
                a["fp"] += fp
                a["fn"] += fn
                a["gt_pos"] += int(gt_pos)
                a["gt_neg"] += int(not gt_pos)
                a["det_pos"] += int(gt_pos and pred_vox > 0)
                a["det_neg"] += int((not gt_pos) and pred_vox > 0)
                a["sum_dice"] += dice
                a["sum_prec"] += prec
                a["sum_rec"] += rec
                a["sum_fp_cc"] += int(fp_cc)
                a["sum_fp_vox"] += int(fp)
                a["n"] += 1

                # write per-case line (only for the first threshold to keep size bounded)
                if thr == thr_list[0]:
                    row_out = {
                        **meta_to_dict(meta),
                        "task": "inference",
                        "case_id": case_id,
                        "split": split,
                        "patch_size": list(ps),
                        "overlap": float(overlap),
                        "tta": tta,
                        "temperature": float(temp),
                        "threshold": thr_f,
                        "min_size": int(min_size),
                        "cc_score": cc_score,
                        "cc_score_thr": float(cc_score_thr),
                        "logits": logits_stats,
                        "prob": prob_stats,
                        "gt_vox": gt_vox,
                        "pred_vox": pred_vox,
                        "tp_vox": tp,
                        "fp_vox": fp,
                        "fn_vox": fn,
                        "voxel_precision": prec,
                        "voxel_recall": rec,
                        "dice": dice,
                        "fp_cc": int(fp_cc),
                        "fp_cc_vox": int(fp_cc_vox),
                    }
                    f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

            if not quiet:
                print(f"[infer] {split} {i+1}/{len(ds)} {case_id} gt_vox={gt_vox}", flush=True)

            gc.collect()

    # aggregate summary per threshold
    per_threshold: list[dict[str, Any]] = []
    for thr in thr_list:
        a = agg[float(thr)]
        n = max(1, int(a["n"]))
        mean_dice = float(a["sum_dice"] / n)
        mean_prec = float(a["sum_prec"] / n)
        mean_rec = float(a["sum_rec"] / n)
        det_rate_case = float(a["det_pos"] / max(1, int(a["gt_pos"]))) if a["gt_pos"] else None
        false_alarm_rate_case = float(a["det_neg"] / max(1, int(a["gt_neg"]))) if a["gt_neg"] else None
        per_threshold.append(
            {
                "threshold": float(thr),
                "mean_dice": mean_dice,
                "voxel_precision": mean_prec,
                "voxel_recall": mean_rec,
                "detection_rate_case": det_rate_case,
                "false_alarm_rate_case": false_alarm_rate_case,
                "mean_fp_cc": float(a["sum_fp_cc"] / n),
                "mean_fp_vox": float(a["sum_fp_vox"] / n),
                "n_cases": int(a["n"]),
                "n_gt_pos": int(a["gt_pos"]),
                "n_gt_neg": int(a["gt_neg"]),
            }
        )

    summary = {
        "meta": meta_to_dict(meta),
        "task": "inference",
        "split": split,
        "thresholds": [float(t) for t in thr_list],
        "per_threshold": per_threshold,
        "settings": {
            "model_path": str(mp),
            "csv_path": str(cp),
            "root": str(data_root),
            "patch_size": list(ps),
            "overlap": float(overlap),
            "tta": tta,
            "temperature": float(temp),
            "min_size": int(min_size),
            "cc_score": cc_score,
            "cc_score_thr": float(cc_score_thr),
            "normalize": normalize,
        },
        "outputs": {
            "per_case_metrics": str(per_case_path),
            "prob_maps_dir": str(prob_maps_dir) if save_prob_maps else None,
        },
    }
    write_json(out_dir / "summary.json", summary)

    # A small visualization for probability distribution
    if prob_samples:
        save_hist_png(
            out_dir / "prob_hist.png",
            np.array(prob_samples, dtype=np.float32),
            title=f"{split}: prob histogram (sampled)",
            xlabel="prob",
            bins=60,
            logy=True,
        )

    md = "\n".join(
        [
            f"# Inference Log ({meta.run_id})",
            "",
            f"- split: `{split}`",
            f"- thresholds: {', '.join(str(t) for t in thr_list)}",
            f"- min_size: {min_size}, cc_score: {cc_score}({cc_score_thr})",
            "",
            "## Outputs",
            f"- `{out_dir / 'summary.json'}`",
            f"- `{per_case_path}`",
            f"- `{out_dir / 'prob_hist.png'}`",
        ]
    )
    write_text(out_dir / "report.md", md + "\n")


if __name__ == "__main__":
    app()
