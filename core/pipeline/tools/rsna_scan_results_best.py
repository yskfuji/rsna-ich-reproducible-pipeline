from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer


@dataclass(frozen=True)
class RunScore:
    run_dir: Path
    best_val_wlogloss: float
    ckpt: Path
    epoch: int | None


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _best_from_log(log_path: Path) -> tuple[float, int | None]:
    best = float("inf")
    best_epoch: int | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            r = json.loads(s)
        except Exception:
            continue
        v = r.get("val_logloss_weighted")
        if not _finite(v):
            continue
        fv = float(v)
        if fv < best:
            best = fv
            ep = r.get("epoch")
            try:
                best_epoch = int(ep) if ep is not None else None
            except Exception:
                best_epoch = None
    return best, best_epoch


app = typer.Typer(add_completion=False)


@app.command()
def main(
    results_dir: Path = typer.Option(Path("results"), help="Directory to scan recursively for run dirs."),
    topk: int = typer.Option(15, help="How many runs to print."),
) -> None:
    root = results_dir.expanduser().resolve()
    if not root.exists():
        raise typer.BadParameter(f"Missing: {root}")

    scores: list[RunScore] = []
    for log_path in root.rglob("log.jsonl"):
        run_dir = log_path.parent
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        best, best_epoch = _best_from_log(log_path)
        if not math.isfinite(best):
            continue
        ckpt = run_dir / "best_wlogloss.pt"
        if not ckpt.exists():
            ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            continue
        scores.append(RunScore(run_dir=run_dir, best_val_wlogloss=float(best), ckpt=ckpt, epoch=best_epoch))

    scores.sort(key=lambda s: s.best_val_wlogloss)
    print(f"found_runs={len(scores)}")
    for i, s in enumerate(scores[: int(topk)], start=1):
        ep = "?" if s.epoch is None else str(s.epoch)
        print(f"{i:02d} best_wlogloss={s.best_val_wlogloss:.6f} epoch={ep} dir={s.run_dir} ckpt={s.ckpt}")


if __name__ == "__main__":
    app()
