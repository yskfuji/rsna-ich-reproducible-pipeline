from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

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
from tools.run_meta import meta_to_dict, write_json, write_text

from src.datasets.isles_dataset import IslesVolumeDataset
from src.models.unet_3d import UNet3D
from src.training.utils_train import prepare_device

import torch

app = typer.Typer(add_completion=False)


def _coerce_option_default(v: Any) -> Any:
    return v.default if isinstance(v, OptionInfo) else v


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


def _dice(tp: int, fp: int, fn: int) -> float:
    return float((2 * tp) / (2 * tp + fp + fn + 1e-8))


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
    threshold: float = typer.Option(0.5, help="prob threshold"),
    # ON settings
    on_min_size: int = typer.Option(20, help="postprocess ON: min_size"),
    on_cc_score: str = typer.Option("none", help="postprocess ON: none|max_prob|p95_prob|mean_prob"),
    on_cc_score_thr: float = typer.Option(0.5, help="postprocess ON: cc score thr"),
    temperature: str = typer.Option("1.0", help="float or from_run_best/from_run_last"),
    normalize: str = typer.Option("nonzero_zscore", help="legacy_zscore|nonzero_zscore|none"),
    allow_missing_label: bool = typer.Option(False, help="missing label -> all zero"),
    tta: str = typer.Option("full", help="full|flip|none"),
    quiet: bool = typer.Option(False, help="suppress per-case prints"),
):
    # When called as a Python function (not via Typer CLI), defaults may be OptionInfo.
    on_min_size = int(_coerce_option_default(on_min_size) or 0)
    on_cc_score = str(_coerce_option_default(on_cc_score))
    on_cc_score_thr = float(_coerce_option_default(on_cc_score_thr))
    temperature = str(_coerce_option_default(temperature))
    allow_missing_label = bool(_coerce_option_default(allow_missing_label))
    quiet = bool(_coerce_option_default(quiet))

    repo_root = Path(__file__).resolve().parents[1]
    mp = Path(model_path).expanduser().resolve()
    cp = Path(csv_path).expanduser().resolve() if Path(csv_path).expanduser().is_absolute() else (repo_root / csv_path).resolve()
    data_root = Path(root).expanduser().resolve() if Path(root).expanduser().is_absolute() else (repo_root / root).resolve()

    run_id_opt = run_id.strip() or None
    cfg_obj: dict[str, Any] = {
        "task": "postprocess_ablation",
        "model_path": str(mp),
        "csv_path": str(cp),
        "root": str(data_root),
        "split": split,
        "patch_size": patch_size,
        "overlap": float(overlap),
        "threshold": float(threshold),
        "postprocess_off": {"min_size": 0, "cc_score": "none", "cc_score_thr": 0.0},
        "postprocess_on": {"min_size": int(on_min_size), "cc_score": on_cc_score, "cc_score_thr": float(on_cc_score_thr)},
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
        dataset_hash_mode=dataset_hash_mode,
    )

    out_dir = run_dir / "postprocess_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config_used.json", {"meta": meta_to_dict(meta), "config": cfg_obj})

    ps: Tuple[int, int, int] = _parse_patch_size(patch_size)
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

    per_case_path = out_dir / "per_case.jsonl"

    rows: list[dict[str, Any]] = []
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
            prob = (1.0 / (1.0 + np.exp(-(logits / float(temp))))).astype(np.float32)

            pred_off = (prob >= float(threshold)).astype(np.uint8)
            pred_on = _filter_min_size(pred_off, int(on_min_size))
            pred_on = _filter_cc_score(pred_on, prob, on_cc_score, float(on_cc_score_thr))

            def _counts(p: np.ndarray):
                tp = int(((p > 0) & (gt > 0)).sum())
                fp = int(((p > 0) & (gt == 0)).sum())
                fn = int(((p == 0) & (gt > 0)).sum())
                return tp, fp, fn

            tp0, fp0, fn0 = _counts(pred_off)
            tp1, fp1, fn1 = _counts(pred_on)

            dice0 = _dice(tp0, fp0, fn0)
            dice1 = _dice(tp1, fp1, fn1)

            rec = {
                **meta_to_dict(meta),
                "task": "postprocess_ablation",
                "split": split,
                "case_id": case_id,
                "threshold": float(threshold),
                "temperature": float(temp),
                "post_off": {"min_size": 0, "cc_score": "none"},
                "post_on": {"min_size": int(on_min_size), "cc_score": on_cc_score, "cc_score_thr": float(on_cc_score_thr)},
                "gt_vox": int(gt.sum()),
                "off": {"pred_vox": int(pred_off.sum()), "tp": tp0, "fp": fp0, "fn": fn0, "dice": dice0},
                "on": {"pred_vox": int(pred_on.sum()), "tp": tp1, "fp": fp1, "fn": fn1, "dice": dice1},
                "delta_dice": float(dice1 - dice0),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

            if not quiet:
                print(f"[post_ablate] {split} {i+1}/{len(ds)} {case_id} dice_off={dice0:.4f} dice_on={dice1:.4f}", flush=True)

    # summary + top diffs
    rows_sorted = sorted(rows, key=lambda r: r["delta_dice"], reverse=True)
    top_improve = rows_sorted[:10]
    top_worse = list(reversed(rows_sorted[-10:]))

    mean_off = float(np.mean([r["off"]["dice"] for r in rows])) if rows else None
    mean_on = float(np.mean([r["on"]["dice"] for r in rows])) if rows else None

    summary = {
        "meta": meta_to_dict(meta),
        "task": "postprocess_ablation",
        "split": split,
        "threshold": float(threshold),
        "postprocess_on": {"min_size": int(on_min_size), "cc_score": on_cc_score, "cc_score_thr": float(on_cc_score_thr)},
        "mean_dice_off": mean_off,
        "mean_dice_on": mean_on,
        "delta_mean_dice": None if (mean_off is None or mean_on is None) else float(mean_on - mean_off),
        "top_improve": [{"case_id": r["case_id"], "delta_dice": r["delta_dice"], "off": r["off"], "on": r["on"]} for r in top_improve],
        "top_worse": [{"case_id": r["case_id"], "delta_dice": r["delta_dice"], "off": r["off"], "on": r["on"]} for r in top_worse],
        "outputs": {"per_case": str(per_case_path)},
    }
    write_json(out_dir / "summary.json", summary)

    def _fmt_table(items: list[dict[str, Any]]) -> list[str]:
        lines = ["|case_id|delta_dice|dice_off|dice_on|gt_vox|pred_off|pred_on|", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in items:
            lines.append(
                "|{case_id}|{dd:+.4f}|{d0:.4f}|{d1:.4f}|{gt}|{p0}|{p1}|".format(
                    case_id=r["case_id"],
                    dd=float(r["delta_dice"]),
                    d0=float(r["off"]["dice"]),
                    d1=float(r["on"]["dice"]),
                    gt=int(r["gt_vox"]),
                    p0=int(r["off"]["pred_vox"]),
                    p1=int(r["on"]["pred_vox"]),
                )
            )
        return lines

    md = "\n".join(
        [
            f"# Postprocess Ablation ({meta.run_id})",
            "",
            f"- split: `{split}`",
            f"- threshold: `{float(threshold)}`",
            f"- ON: min_size={int(on_min_size)}, cc_score={on_cc_score}({float(on_cc_score_thr)})",
            f"- mean_dice_off: `{mean_off}`",
            f"- mean_dice_on: `{mean_on}`",
            "",
            "## Top Improve",
            *(_fmt_table(top_improve) if top_improve else ["(none)"]),
            "",
            "## Top Worse",
            *(_fmt_table(top_worse) if top_worse else ["(none)"]),
            "",
            "## Outputs",
            f"- `{out_dir / 'summary.json'}`",
            f"- `{per_case_path}`",
        ]
    )
    write_text(out_dir / "diff.md", md + "\n")


if __name__ == "__main__":
    app()
