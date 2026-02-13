from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

app = typer.Typer(add_completion=False)


def _load_yaml(p: Path) -> dict[str, Any]:
    return yaml.safe_load(p.read_text())


def _dump_yaml(obj: dict[str, Any]) -> str:
    return yaml.safe_dump(obj, sort_keys=False)


@app.command()
def main(
    *,
    base_config: Path = typer.Option(..., help="Base training YAML to copy (e.g. recipe yaml)."),
    kfold_dir: Path = typer.Option(..., help="Directory with fold0.csv..fold{K-1}.csv"),
    out_dir: Path = typer.Option(..., help="Where to write generated fold YAMLs"),
    data_root: Path = typer.Option(..., help="Processed dataset root to use (images/, labels/)."),
    exp_prefix: str = typer.Option("", help="Override experiment_name prefix (default: base experiment_name)."),
    k: int = typer.Option(5, help="Number of folds"),
    strip_init_from: bool = typer.Option(
        True,
        help="Remove train.init_from from generated configs (default: True; keeps k-fold fair unless you opt-in).",
    ),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _load_yaml(base_config)
    base_exp = str(base.get("experiment_name") or base_config.stem)
    prefix = exp_prefix.strip() or base_exp

    written: list[str] = []

    for fold in range(int(k)):
        fold_csv = kfold_dir / f"fold{fold}.csv"
        if not fold_csv.exists():
            raise FileNotFoundError(f"Missing: {fold_csv}")

        cfg = json.loads(json.dumps(base))  # deep copy (yaml may contain dict/list)

        cfg["experiment_name"] = f"{prefix}_kfold{int(k)}_f{fold}_ts222"

        cfg.setdefault("data", {})
        cfg["data"]["csv_path"] = str(fold_csv.as_posix())
        cfg["data"]["root"] = str(data_root.as_posix())

        cfg.setdefault("train", {})
        if strip_init_from:
            cfg["train"].pop("init_from", None)

        # If data is already resampled to isotropic spacing, disable evaluation-style upsample.
        if "resample_max_zoom_mm" in cfg["data"]:
            cfg["data"]["resample_max_zoom_mm"] = 0.0

        out_path = out_dir / f"{cfg['experiment_name']}.yaml"
        out_path.write_text(_dump_yaml(cfg), encoding="utf-8")
        written.append(str(out_path))

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "base_config": str(base_config),
                "kfold_dir": str(kfold_dir),
                "data_root": str(data_root),
                "k": int(k),
                "written": written,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"wrote {len(written)} configs to {out_dir}")


if __name__ == "__main__":
    app()
