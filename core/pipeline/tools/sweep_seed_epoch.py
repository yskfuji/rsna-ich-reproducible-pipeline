"""Generate training configs sweeping seed and epochs.

Usage example:
  python -m tools.sweep_seed_epoch --base configs/...yaml --out-dir configs/generated/_seed_epoch_... \
    --seeds 1,2,3 --epochs 20,40 --suffix pr2pw4bw0p5
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False)


def _csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(float(p)))
    return out


def _slug(s: str) -> str:
    s = str(s)
    s = s.replace(".", "p").replace("-", "m")
    s = "".join(ch for ch in s if ch.isalnum() or ch in {"_", "p", "m"})
    return s


@dataclass(frozen=True)
class Item:
    exp: str
    path: str
    seed: int
    epochs: int


@app.command()
def main(
    base: str = typer.Option(..., help="base YAML config"),
    out_dir: str = typer.Option(..., help="output directory to write generated YAMLs"),
    seeds: str = typer.Option("42", help="comma-separated seeds"),
    epochs: str = typer.Option("20", help="comma-separated epochs"),
    suffix: str = typer.Option("", help="optional suffix to append to experiment_name"),
) -> None:
    base_p = Path(base)
    cfg0 = yaml.safe_load(base_p.read_text())

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    seed_list = _csv_ints(seeds)
    epoch_list = _csv_ints(epochs)

    items: list[Item] = []

    base_exp = str(cfg0.get("experiment_name", base_p.stem))
    suf = _slug(suffix) if suffix else ""

    for e in epoch_list:
        for sd in seed_list:
            cfg = deepcopy(cfg0)
            cfg.setdefault("train", {})
            cfg["train"]["seed"] = int(sd)
            cfg["train"]["epochs"] = int(e)

            exp = base_exp
            if suf:
                exp = f"{exp}_{suf}"
            exp = f"{exp}_e{int(e)}_s{int(sd)}"
            cfg["experiment_name"] = exp

            out_yaml = out_p / f"{exp}.yaml"
            out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
            items.append(Item(exp=exp, path=str(out_yaml), seed=int(sd), epochs=int(e)))

    index = {
        "base": str(base_p),
        "out_dir": str(out_p),
        "seeds": seed_list,
        "epochs": epoch_list,
        "suffix": suffix,
        "items": [i.__dict__ for i in items],
    }
    index_path = out_p / "seed_epoch_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")

    typer.echo(f"wrote {len(items)} configs to {out_p}")
    typer.echo(f"index: {index_path}")


if __name__ == "__main__":
    app()
