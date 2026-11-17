from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    arch: str
    device: str
    stack_slices: int
    image_size: int
    batch_size: int
    num_workers: int
    n_train: int
    n_val: int
    best_val_loss: float
    best_val_loss_epoch: int
    last_val_loss: float
    best_val_loss_plain: float
    best_val_loss_plain_epoch: int
    last_val_loss_plain: float
    best_val_wlogloss: float
    best_val_wlogloss_epoch: int
    last_val_wlogloss: float
    best_val_auc_mean: float
    best_val_auc_mean_epoch: int
    last_val_auc_mean: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _summarize_run(run_dir: Path) -> RunSummary:
    meta = _read_json(run_dir / "meta.json")
    logs = _read_jsonl(run_dir / "log.jsonl")
    if not logs:
        raise ValueError(f"No log.jsonl rows: {run_dir}")

    def _to_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _to_int(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return -1

    val_losses = np.asarray([_to_float(r.get("val_loss")) for r in logs], dtype=np.float64)
    val_losses_plain = np.asarray([
        _to_float(r.get("val_loss_plain", r.get("val_loss"))) for r in logs
    ], dtype=np.float64)
    val_wlogloss = np.asarray([
        _to_float(r.get("val_logloss_weighted")) for r in logs
    ], dtype=np.float64)
    val_aucs = np.asarray([_to_float(r.get("val_auc_mean")) for r in logs], dtype=np.float64)
    epochs = np.asarray([_to_int(r.get("epoch")) for r in logs], dtype=np.int64)

    best_loss_idx = int(np.nanargmin(val_losses)) if np.isfinite(val_losses).any() else 0
    best_loss_plain_idx = int(np.nanargmin(val_losses_plain)) if np.isfinite(val_losses_plain).any() else 0
    best_wlogloss_idx = int(np.nanargmin(val_wlogloss)) if np.isfinite(val_wlogloss).any() else 0
    best_auc_idx = int(np.nanargmax(val_aucs)) if np.isfinite(val_aucs).any() else 0

    return RunSummary(
        run_dir=run_dir,
        arch=str(meta.get("arch", "cnn")),
        device=str(meta.get("device", "")),
        stack_slices=int(meta.get("stack_slices", 1)),
        image_size=int(meta.get("image_size", 0)),
        batch_size=int(meta.get("batch_size", 0)),
        num_workers=int(meta.get("num_workers", 0)),
        n_train=int(meta.get("n_train", 0)),
        n_val=int(meta.get("n_val", 0)),
        best_val_loss=float(val_losses[best_loss_idx]),
        best_val_loss_epoch=int(epochs[best_loss_idx]),
        last_val_loss=float(val_losses[-1]),
        best_val_loss_plain=float(val_losses_plain[best_loss_plain_idx]),
        best_val_loss_plain_epoch=int(epochs[best_loss_plain_idx]),
        last_val_loss_plain=float(val_losses_plain[-1]),
        best_val_wlogloss=float(val_wlogloss[best_wlogloss_idx]),
        best_val_wlogloss_epoch=int(epochs[best_wlogloss_idx]),
        last_val_wlogloss=float(val_wlogloss[-1]),
        best_val_auc_mean=float(val_aucs[best_auc_idx]),
        best_val_auc_mean_epoch=int(epochs[best_auc_idx]),
        last_val_auc_mean=float(val_aucs[-1]),
    )


def _fmt(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.6f}"


@app.command()
def main(
    runs: list[Path] = typer.Argument(..., help="One or more run directories (must contain meta.json + log.jsonl)"),
):
    summaries: list[RunSummary] = []
    for r in runs:
        rd = r.expanduser().resolve()
        summaries.append(_summarize_run(rd))

    # stable order by best_val_loss_plain
    summaries.sort(
        key=lambda s: (
            (np.inf if not np.isfinite(s.best_val_wlogloss) else s.best_val_wlogloss),
            (np.inf if not np.isfinite(s.best_val_loss_plain) else s.best_val_loss_plain),
        )
    )

    print(
        "| run_dir | arch | device | stack | img | bs | nw | best_val_wlogloss (ep) | last_val_wlogloss | best_val_loss_plain (ep) | last_val_loss_plain | best_val_auc_mean (ep) | last_val_auc_mean |",
        flush=True,
    )
    print(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        flush=True,
    )
    for s in summaries:
        print(
            "| "
            + " | ".join(
                [
                    s.run_dir.name,
                    s.arch,
                    s.device,
                    str(s.stack_slices),
                    str(s.image_size),
                    str(s.batch_size),
                    str(s.num_workers),
                    f"{_fmt(s.best_val_wlogloss)} ({s.best_val_wlogloss_epoch})",
                    _fmt(s.last_val_wlogloss),
                    f"{_fmt(s.best_val_loss_plain)} ({s.best_val_loss_plain_epoch})",
                    _fmt(s.last_val_loss_plain),
                    f"{_fmt(s.best_val_auc_mean)} ({s.best_val_auc_mean_epoch})",
                    _fmt(s.last_val_auc_mean),
                ]
            )
            + " |",
            flush=True,
        )


if __name__ == "__main__":
    app()
