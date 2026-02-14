"""Grid-search postprocessing (min_size/top_k/cc_score) + threshold sweep on a fixed model.

This is a thin wrapper around `python -m src.evaluation.evaluate_isles` that runs multiple
configurations and aggregates the best threshold per configuration.

Designed for: test-side optimization to quantify how much is left on the table.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer

app = typer.Typer(add_completion=False)


def _csv_floats(s: str) -> list[float]:
    out: list[float] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(float(p)))
    return out


def _slug(v: object) -> str:
    s = str(v)
    s = s.replace(".", "p").replace("-", "m")
    s = "".join(ch for ch in s if ch.isalnum() or ch in {"_", "p", "m"})
    return s


@dataclass(frozen=True)
class SweepRow:
    out_dir: str
    resample_max_zoom_mm: float
    min_size: int
    top_k: int
    cc_score: str
    cc_score_thr: float
    best_threshold: float
    focus_bucket: str
    focus_mean_dice: float | None
    mean_dice: float | None
    median_dice: float | None
    voxel_precision: float | None
    voxel_recall: float | None
    detection_rate_case: float | None
    false_alarm_rate_case: float | None
    mean_fp_cc: float | None
    mean_fp_vox: float | None
    mean_pred_vox: float | None


def _run_eval(
    *,
    python_exe: str,
    env: dict[str, str],
    model_path: str | None,
    probs_dir: str | None,
    csv_path: str,
    root: str,
    split: str,
    out_dir: Path,
    patch_size: str,
    overlap: float,
    thresholds: str,
    min_size: int,
    top_k: int,
    cc_score: str,
    cc_score_thr: float,
    normalize: str,
    tta: str,
    resample_max_zoom_mm: float,
    slice_spacing_source: str,
    quiet: bool,
) -> None:
    cmd = [python_exe, "-m", "src.evaluation.evaluate_isles"]
    if probs_dir is not None:
        cmd += ["--probs-dir", probs_dir]
    else:
        if not model_path:
            raise ValueError("model_path must be set when probs_dir is None")
        cmd += ["--model-path", model_path]

    cmd += [
        "--csv-path",
        csv_path,
        "--root",
        root,
        "--split",
        split,
        "--out-dir",
        str(out_dir),
        "--patch-size",
        patch_size,
        "--overlap",
        str(overlap),
        "--thresholds",
        thresholds,
        "--min-size",
        str(int(min_size)),
        "--top-k",
        str(int(top_k)),
        "--cc-score",
        str(cc_score),
        "--cc-score-thr",
        str(float(cc_score_thr)),
        "--normalize",
        normalize,
        "--tta",
        tta,
        "--resample-max-zoom-mm",
        str(float(resample_max_zoom_mm)),
        "--slice-spacing-source",
        str(slice_spacing_source),
    ]
    if quiet:
        cmd.append("--quiet")

    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, env=env, check=True)


def _best_from_summary(summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    per = s.get("per_threshold") or []
    if not per:
        raise ValueError(f"per_threshold missing/empty in {summary_path}")
    best = max(per, key=lambda r: (r.get("mean_dice") is not None, float(r.get("mean_dice") or -1.0)))
    return best


def _best_from_summary_focus(summary_path: Path, *, focus_bucket: str) -> tuple[dict, float | None]:
    """Pick threshold row by focus bucket mean_dice.

    focus_bucket:
      - 'all': use global mean_dice
      - otherwise: use per_threshold[i].by_slice_spacing[focus_bucket].mean_dice
    """
    s = json.loads(summary_path.read_text())
    per = s.get("per_threshold") or []
    if not per:
        raise ValueError(f"per_threshold missing/empty in {summary_path}")

    fb = str(focus_bucket).strip()
    if fb in {"", "all"}:
        best = max(per, key=lambda r: (r.get("mean_dice") is not None, float(r.get("mean_dice") or -1.0)))
        return best, best.get("mean_dice")

    def score(r: dict) -> float:
        by = r.get("by_slice_spacing") or {}
        sub = by.get(fb) or {}
        v = sub.get("mean_dice")
        return float(v) if v is not None else -1.0

    best = max(per, key=score)
    by = best.get("by_slice_spacing") or {}
    sub = by.get(fb) or {}
    return best, sub.get("mean_dice")


@app.command()
def main(
    model_path: str = typer.Option("", help="model checkpoint (required unless --probs-dir is set)"),
    probs_dir: str = typer.Option("", help="directory with <case_id>.npz probs (required unless --model-path is set)"),
    csv_path: str = typer.Option(...),
    root: str = typer.Option(...),
    split: str = typer.Option("test"),
    out_root: str = typer.Option(..., help="root directory to write sweep outputs"),
    patch_size: str = typer.Option("48,48,24"),
    overlap: float = typer.Option(0.5),
    normalize: str = typer.Option("nonzero_zscore"),
    tta: str = typer.Option("full"),
    thresholds: str = typer.Option(
        "0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.40,0.45,0.50,0.60",
        help="comma-separated thresholds to evaluate per configuration",
    ),
    min_sizes: str = typer.Option("0,20"),
    top_ks: str = typer.Option("0,1"),
    cc_scores: str = typer.Option("none,max_prob"),
    cc_score_thrs: str = typer.Option("0.5,0.7"),
    resamples: str = typer.Option("0.0", help="comma-separated resample_max_zoom_mm values (e.g., 0,2.0)"),
    slice_spacing_source: str = typer.Option(
        "raw",
        help=(
            "How to compute slice-spacing buckets when resampling is enabled. "
            "Use 'raw' to keep original bucket membership (recommended for comparing resamples)."
        ),
    ),
    focus_bucket: str = typer.Option(
        "all",
        help=(
            "Optimization target bucket for selecting best threshold per configuration. "
            "Use 'all' for global mean_dice, or a by_slice_spacing key like 'le_3mm'/'gt_3mm'."
        ),
    ),
    python_exe: str = typer.Option(sys.executable, help="python executable to use for subprocess eval"),
    quiet: bool = typer.Option(True),
) -> None:
    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)

    model_path = str(model_path).strip()
    probs_dir = str(probs_dir).strip()
    if bool(model_path) == bool(probs_dir):
        raise ValueError("Provide exactly one of --model-path or --probs-dir")

    env = os.environ.copy()
    # Ensure src.* imports resolve when invoked via subprocess
    env.setdefault("PYTHONPATH", str(Path.cwd()))

    ms_list = _csv_ints(min_sizes)
    tk_list = _csv_ints(top_ks)
    cc_list = [c.strip() for c in str(cc_scores).split(",") if c.strip()]
    cct_list = _csv_floats(cc_score_thrs)
    rs_list = _csv_floats(resamples)
    fb = str(focus_bucket).strip() or "all"

    rows: list[SweepRow] = []

    for rs in rs_list:
        for ms in ms_list:
            for tk in tk_list:
                for cc in cc_list:
                    cc_norm = cc.strip().lower()
                    thrs_iter: Iterable[float]
                    if cc_norm in {"none", "off", "false"}:
                        thrs_iter = [float(cct_list[0] if cct_list else 0.5)]
                    else:
                        thrs_iter = cct_list

                    for cct in thrs_iter:
                        name = f"rs{_slug(rs)}_ms{_slug(ms)}_tk{_slug(tk)}_cc{_slug(cc_norm)}_cct{_slug(cct)}"
                        out_dir = out_root_p / name
                        summary_path = out_dir / "summary.json"

                        if not summary_path.exists():
                            _run_eval(
                                python_exe=python_exe,
                                env=env,
                                model_path=model_path if model_path else None,
                                probs_dir=probs_dir if probs_dir else None,
                                csv_path=csv_path,
                                root=root,
                                split=split,
                                out_dir=out_dir,
                                patch_size=patch_size,
                                overlap=overlap,
                                thresholds=thresholds,
                                min_size=ms,
                                top_k=tk,
                                cc_score=cc_norm,
                                cc_score_thr=float(cct),
                                normalize=normalize,
                                tta=tta,
                                resample_max_zoom_mm=float(rs),
                                slice_spacing_source=slice_spacing_source,
                                quiet=quiet,
                            )

                        best, focus_md = _best_from_summary_focus(summary_path, focus_bucket=fb)

                        rows.append(
                            SweepRow(
                                out_dir=name,
                                resample_max_zoom_mm=float(rs),
                                min_size=int(ms),
                                top_k=int(tk),
                                cc_score=str(cc_norm),
                                cc_score_thr=float(cct),
                                best_threshold=float(best.get("threshold")),
                                focus_bucket=fb,
                                focus_mean_dice=focus_md,
                                mean_dice=best.get("mean_dice"),
                                median_dice=best.get("median_dice"),
                                voxel_precision=best.get("voxel_precision"),
                                voxel_recall=best.get("voxel_recall"),
                                detection_rate_case=best.get("detection_rate_case"),
                                false_alarm_rate_case=best.get("false_alarm_rate_case"),
                                mean_fp_cc=best.get("mean_fp_cc"),
                                mean_fp_vox=best.get("mean_fp_vox"),
                                mean_pred_vox=best.get("mean_pred_vox"),
                            )
                        )

    # write aggregate
    cols = [
        "out_dir",
        "resample_max_zoom_mm",
        "min_size",
        "top_k",
        "cc_score",
        "cc_score_thr",
        "best_threshold",
        "focus_bucket",
        "focus_mean_dice",
        "mean_dice",
        "median_dice",
        "voxel_precision",
        "voxel_recall",
        "detection_rate_case",
        "false_alarm_rate_case",
        "mean_fp_cc",
        "mean_fp_vox",
        "mean_pred_vox",
    ]
    out_tsv = out_root_p / "sweep.tsv"
    lines = ["\t".join(cols)]
    def _sort_key(x: SweepRow):
        # primary: focus bucket dice
        md = x.focus_mean_dice
        return (md is None, -(md or -1.0), x.mean_fp_vox or 1e18)

    for r in sorted(rows, key=_sort_key):
        d = r.__dict__
        lines.append("\t".join(str(d.get(c)) for c in cols))
    out_tsv.write_text("\n".join(lines) + "\n")

    # print top-10
    typer.echo(f"wrote {out_tsv}")
    typer.echo(f"TOP (by focus_mean_dice={fb}, tie-break FPvox):")
    for r in sorted(rows, key=_sort_key)[:10]:
        typer.echo(
            f"{r.out_dir}\tfocus_mean_dice={r.focus_mean_dice}\tmean_dice={r.mean_dice}\tthr={r.best_threshold}"
            f"\trs={r.resample_max_zoom_mm}\tms={r.min_size}\ttk={r.top_k}\tcc={r.cc_score}@{r.cc_score_thr}"
        )


if __name__ == "__main__":
    app()
