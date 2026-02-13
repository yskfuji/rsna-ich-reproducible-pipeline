from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)


def _parse_float_list(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [float(p) for p in parts]


def _safe_slug(x: str) -> str:
    return (
        str(x)
        .strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace(":", "-")
        .replace("=", "-")
        .replace(".", "p")
    )


def _write_yaml(path: Path, obj: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


@app.command()
def generate(
    base_config: str = typer.Option(..., help="Base training config (YAML)"),
    out_dir: str = typer.Option("configs/generated", help="Where to write generated YAML configs"),
    modes: str = typer.Option(
        "dice_bce,dice_focal,tversky",
        help="Comma-separated loss modes to generate: dice_bce|dice_focal|tversky",
    ),
    # dice_bce
    pos_weights: str = typer.Option("1,2,4,8", help="dice_bce pos_weight grid (comma-separated)"),
    bce_weights: str = typer.Option("0.25,0.5,1.0", help="dice_bce bce_weight grid (comma-separated)"),
    # dice_focal
    focal_alphas: str = typer.Option("0.25", help="dice_focal alpha grid"),
    focal_gammas: str = typer.Option("2.0", help="dice_focal gamma grid"),
    # tversky
    tversky_alphas: str = typer.Option("0.3,0.4", help="tversky alpha grid"),
    tversky_betas: str = typer.Option("0.6,0.7", help="tversky beta grid"),
    smooth: float = typer.Option(1.0, help="smooth used for dice/tversky"),
    suffix: str = typer.Option("pr2", help="suffix added to experiment_name"),
):
    """Generate a grid of training configs for PR-2 loss sweeps.

    This does not run training; it only writes YAMLs you can execute with:
      python -m src.training.train_3d_unet --config <generated.yaml>
    """

    repo_root = Path(__file__).resolve().parents[1]
    base_path = Path(base_config).expanduser()
    if not base_path.is_absolute():
        base_path = (repo_root / base_path).resolve()

    import yaml

    base = yaml.safe_load(base_path.read_text())
    if not isinstance(base, dict):
        raise ValueError("base_config must be a YAML mapping")

    exp0 = str(base.get("experiment_name") or "exp")
    out_root = Path(out_dir)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()

    mode_list = [m.strip().lower() for m in str(modes).split(",") if m.strip()]
    generated: list[dict[str, Any]] = []

    # dice_bce
    if "dice_bce" in mode_list:
        for pw in _parse_float_list(pos_weights):
            for bw in _parse_float_list(bce_weights):
                cfg = deepcopy(base)
                cfg.setdefault("train", {})
                cfg["train"]["loss"] = "dice_bce"
                lp = dict(cfg["train"].get("loss_params") or {})
                lp.update({"smooth": float(smooth), "pos_weight": float(pw), "bce_weight": float(bw)})
                cfg["train"]["loss_params"] = lp

                tag = f"{suffix}_dice_bce_pw{_safe_slug(pw)}_bw{_safe_slug(bw)}"
                cfg["experiment_name"] = f"{exp0}_{tag}"

                out_p = out_root / f"{cfg['experiment_name']}.yaml"
                _write_yaml(out_p, cfg)
                generated.append({"path": str(out_p), "train.loss": "dice_bce", "pos_weight": pw, "bce_weight": bw})

    # dice_focal
    if "dice_focal" in mode_list:
        for a in _parse_float_list(focal_alphas):
            for g in _parse_float_list(focal_gammas):
                cfg = deepcopy(base)
                cfg.setdefault("train", {})
                cfg["train"]["loss"] = "dice_focal"
                lp = dict(cfg["train"].get("loss_params") or {})
                lp.update({"smooth": float(smooth), "alpha": float(a), "gamma": float(g)})
                cfg["train"]["loss_params"] = lp

                tag = f"{suffix}_dice_focal_a{_safe_slug(a)}_g{_safe_slug(g)}"
                cfg["experiment_name"] = f"{exp0}_{tag}"

                out_p = out_root / f"{cfg['experiment_name']}.yaml"
                _write_yaml(out_p, cfg)
                generated.append({"path": str(out_p), "train.loss": "dice_focal", "alpha": a, "gamma": g})

    # tversky
    if "tversky" in mode_list:
        for a in _parse_float_list(tversky_alphas):
            for b in _parse_float_list(tversky_betas):
                cfg = deepcopy(base)
                cfg.setdefault("train", {})
                cfg["train"]["loss"] = "tversky"
                lp = dict(cfg["train"].get("loss_params") or {})
                lp.update({"smooth": float(smooth), "alpha": float(a), "beta": float(b)})
                cfg["train"]["loss_params"] = lp

                tag = f"{suffix}_tversky_a{_safe_slug(a)}_b{_safe_slug(b)}"
                cfg["experiment_name"] = f"{exp0}_{tag}"

                out_p = out_root / f"{cfg['experiment_name']}.yaml"
                _write_yaml(out_p, cfg)
                generated.append({"path": str(out_p), "train.loss": "tversky", "alpha": a, "beta": b})

    meta_p = out_root / f"{_safe_slug(exp0)}_{suffix}_loss_sweep_index.json"
    meta_p.write_text(json.dumps({"base_config": str(base_path), "generated": generated}, indent=2))

    print(f"wrote {len(generated)} configs under: {out_root}")
    print(f"index: {meta_p}")


if __name__ == "__main__":
    app()
