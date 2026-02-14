from __future__ import annotations

import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer


@dataclass(frozen=True)
class BestRun:
    run_dir: Path
    best_val_wlogloss: float
    ckpt_path: Path
    arch: str
    image_size: int
    windows: str
    preprocess: str
    stack_slices: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _best_wlogloss_from_logs(logs: list[dict[str, Any]]) -> float:
    best = float("inf")
    for r in logs:
        v = r.get("val_logloss_weighted")
        if _finite(v):
            best = min(best, float(v))
    return best


def _candidate_from_run_dir(run_dir: Path) -> BestRun | None:
    meta_path = run_dir / "meta.json"
    log_path = run_dir / "log.jsonl"
    if not meta_path.exists() or not log_path.exists():
        return None

    try:
        meta = _read_json(meta_path)
        logs = _iter_jsonl(log_path)
        best_ll = _best_wlogloss_from_logs(logs)
    except Exception:
        return None

    ckpt = run_dir / "best_wlogloss.pt"
    if not ckpt.exists():
        ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        return None

    arch = str(meta.get("arch", "resnet18"))
    image_size = int(meta.get("image_size", 256))
    windows = str(meta.get("windows", "40,80;80,200;600,2800"))
    preprocess = str(meta.get("preprocess", "legacy"))
    stack_slices = int(meta.get("stack_slices", 1))

    return BestRun(
        run_dir=run_dir,
        best_val_wlogloss=float(best_ll),
        ckpt_path=ckpt,
        arch=arch,
        image_size=image_size,
        windows=windows,
        preprocess=preprocess,
        stack_slices=stack_slices,
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    runs_dir: Path = typer.Option(..., help="Directory containing multiple run subdirs with meta.json/log.jsonl"),
    fmt: str = typer.Option("shell", help="shell | json"),
) -> None:
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.exists():
        raise typer.BadParameter(f"Missing: {runs_dir}")

    best: BestRun | None = None

    # Allow passing a single run directory (meta.json/log.jsonl at top-level).
    direct = _candidate_from_run_dir(runs_dir)
    if direct is not None:
        best = direct

    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        cand = _candidate_from_run_dir(p)
        if cand is None:
            continue
        if best is None or cand.best_val_wlogloss < best.best_val_wlogloss:
            best = cand

    if best is None:
        raise typer.BadParameter(f"No valid runs found under: {runs_dir}")

    payload = {
        "run_dir": str(best.run_dir),
        "best_val_wlogloss": float(best.best_val_wlogloss),
        "ckpt": str(best.ckpt_path),
        "arch": best.arch,
        "image_size": int(best.image_size),
        "windows": best.windows,
        "preprocess": str(best.preprocess),
        "stack_slices": int(best.stack_slices),
    }

    fmt_s = str(fmt).strip().lower()
    if fmt_s == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if fmt_s != "shell":
        raise typer.BadParameter("fmt must be 'shell' or 'json'")

    # shell-safe enough for our paths (no newlines). Caller uses: eval "$(...)".
    # NOTE: quote values to survive zsh/bash `eval`, especially `windows` contains `;`.
    print(f"BEST_RUN_DIR={shlex.quote(str(payload['run_dir']))}")
    print(f"BEST_VAL_WLOGLOSS={shlex.quote(str(payload['best_val_wlogloss']))}")
    print(f"BEST_CKPT={shlex.quote(str(payload['ckpt']))}")
    print(f"BEST_ARCH={shlex.quote(str(payload['arch']))}")
    print(f"BEST_IMAGE_SIZE={shlex.quote(str(payload['image_size']))}")
    print(f"BEST_WINDOWS={shlex.quote(str(payload['windows']))}")
    print(f"BEST_STACK_SLICES={shlex.quote(str(payload['stack_slices']))}")


if __name__ == "__main__":
    app()
