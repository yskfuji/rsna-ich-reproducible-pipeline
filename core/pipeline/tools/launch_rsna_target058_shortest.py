from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class LaunchConfig:
    rsna_root: Path
    preprocessed_root: Path | None
    device: str
    limit_images: int
    epochs: int
    val_frac: float
    lr: float
    preprocess: str
    out_base: Path
    stack_slices: int
    split_by: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_rsna_root(maybe_rsna_root: str | None) -> Path:
    value = (maybe_rsna_root or os.environ.get("RSNA_ROOT") or "").strip()
    if not value:
        raise SystemExit(
            "Missing RSNA root. Provide --rsna-root or set env RSNA_ROOT to the dataset root "
            "(contains stage_2_train.csv and stage_2_train/; stage_2_test/ and stage_2_sample_submission.csv)."
        )
    return Path(value).expanduser().resolve()


def _resolve_preprocessed_root(maybe: str | None) -> Path | None:
    v = (maybe or os.environ.get("RSNA_PREPROCESSED_ROOT") or "").strip()
    if not v:
        return None
    return Path(v).expanduser().resolve()


def _build_train_cmd(*, arch: str, out_dir: Path, batch_size: int, cfg: LaunchConfig) -> list[str]:
    root = _project_root()
    entry = root / "train_rsna_cnn2d_classifier.py"
    if not entry.exists():
        raise FileNotFoundError(f"Missing entrypoint: {entry}")

    common: list[str] = [
        sys.executable,
        "-u",
        str(entry),
        "train",
        "--rsna-root",
        str(cfg.rsna_root),
        "--limit-images",
        str(int(cfg.limit_images)),
        "--val-frac",
        str(float(cfg.val_frac)),
        "--split-by",
        str(cfg.split_by),
        "--seed",
        "0",
        "--epochs",
        str(int(cfg.epochs)),
        "--lr",
        str(float(cfg.lr)),
        "--weight-decay",
        "1e-4",
        "--windows",
        "40,80;80,200;600,2800",
        "--preprocess",
        str(cfg.preprocess),
        "--aug",
        "--scheduler",
        "--loss-any-weight",
        "2.0",
        "--log-every-steps",
        "200",
        "--image-size",
        "384",
        "--stack-slices",
        str(int(cfg.stack_slices)),
        "--num-workers",
        "0",
        "--out-dir",
        str(out_dir),
        "--arch",
        str(arch),
        "--pretrained",
        "--input-normalize",
        "none",
        "--batch-size",
        str(int(batch_size)),
        "--no-use-pos-weight",
        "--no-use-sampler",
    ]

    if cfg.preprocessed_root is not None:
        common += ["--preprocessed-root", str(cfg.preprocessed_root)]
    else:
        common += ["--cache-dir", str(cfg.out_base / "cache_train")]
    return common


def _run_foreground(cfg: LaunchConfig) -> None:
    root = _project_root()
    env = dict(os.environ)
    env["TORCH_DEVICE"] = cfg.device
    env.setdefault("PYTHONUNBUFFERED", "1")

    cfg.out_base.mkdir(parents=True, exist_ok=True)

    run_tag = "2d" if int(cfg.stack_slices) == 1 else f"25d_stack{int(cfg.stack_slices)}"
    eff_dir = cfg.out_base / f"effb0_{run_tag}_img384_slicesplit_e{cfg.epochs}"
    cnx_dir = cfg.out_base / f"convnext_tiny_{run_tag}_img384_slicesplit_e{cfg.epochs}"

    log_path = cfg.out_base / "launcher.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[start] {datetime.now().isoformat()}\n")
        log.write(f"rsna_root={cfg.rsna_root}\n")
        log.write(f"device={cfg.device} limit_images={cfg.limit_images} epochs={cfg.epochs} val_frac={cfg.val_frac} lr={cfg.lr}\n")
        log.write(f"preprocess={cfg.preprocess}\n")
        log.write(f"stack_slices={cfg.stack_slices} split_by={cfg.split_by}\n")
        log.write(f"out_base={cfg.out_base}\n\n")
        log.flush()

        for name, cmd in [
            ("effb0", _build_train_cmd(arch="efficientnet_b0", out_dir=eff_dir, batch_size=10, cfg=cfg)),
            ("convnext_tiny", _build_train_cmd(arch="convnext_tiny", out_dir=cnx_dir, batch_size=6, cfg=cfg)),
        ]:
            log.write(f"\n[{name}] RUN: {' '.join(cmd)}\n")
            log.flush()
            subprocess.check_call(cmd, cwd=str(root), env=env, stdout=log, stderr=log)

        log.write(f"\n[done] {datetime.now().isoformat()}\n")
        log.flush()

    print(f"[done] runs under: {cfg.out_base}")
    print(f"[next] smoke ensemble:")
    print(
        "MAX_TEST_IMAGES=200 ./scripts/make_rsna_submission_from_two_bests.zsh "
        f"{shlex.quote(str(cfg.rsna_root))} {shlex.quote(str(eff_dir))} {shlex.quote(str(cnx_dir))} "
        f"{shlex.quote('submission_ensemble_smoke.csv')} {shlex.quote(str(cfg.device))}"
    )


def _detach(argv: list[str], *, cfg: LaunchConfig) -> int:
    root = _project_root()
    cfg.out_base.mkdir(parents=True, exist_ok=True)
    log_path = cfg.out_base / "launcher.log"

    # Re-run this script in the background without --detach.
    bg_argv = [a for a in argv if a != "--detach"]
    cmd = [sys.executable, str(Path(__file__).resolve()), *bg_argv]

    log_f = log_path.open("a", encoding="utf-8")
    p = subprocess.Popen(
        cmd,
        cwd=str(root),
        env={**os.environ, "TORCH_DEVICE": cfg.device, "PYTHONUNBUFFERED": "1"},
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,
        close_fds=True,
    )
    # Do not close log_f here; keep FD alive for the child.
    print(f"[launched] pid={p.pid}")
    print(f"out_base={cfg.out_base}")
    print(f"log={log_path}")
    return p.pid


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rsna-root", default=None, type=str, help="RSNA dataset root. If omitted, uses env RSNA_ROOT.")
    p.add_argument(
        "--preprocessed-root",
        default=None,
        type=str,
        help="If set, train from <preprocessed_root>/train.sqlite (DICOM not required). "
        "If --rsna-root is omitted, defaults to <preprocessed_root>/rsna_meta for CSVs.",
    )
    p.add_argument("--device", default="mps", type=str)
    p.add_argument("--limit-images", default=200000, type=int)
    p.add_argument("--epochs", default=6, type=int)
    p.add_argument("--val-frac", default=0.02, type=float)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument(
        "--preprocess",
        default="gpt52",
        type=str,
        help="RSNA DICOM preprocessing mode passed to training: legacy | gpt52 (default: gpt52)",
    )
    p.add_argument(
        "--stack-slices",
        default=1,
        type=int,
        help="2.5D: odd number of slices to stack as channels (1=2D). Use 3 for pseudo-2.5D.",
    )
    p.add_argument(
        "--split-by",
        default=None,
        type=str,
        help="Dataset split unit: slice | series | study. If omitted: slice for 2D, study for 2.5D.",
    )
    p.add_argument(
        "--out-base",
        default="",
        type=str,
        help="Output base dir. Default: results/rsna_target058_shortest_<timestamp>",
    )
    p.add_argument("--detach", action="store_true", help="Launch in background and return immediately")

    ns = p.parse_args(argv)

    root = _project_root()
    preprocessed_root = _resolve_preprocessed_root(ns.preprocessed_root)
    if ns.rsna_root is None and preprocessed_root is not None:
        rsna_root = (preprocessed_root / "rsna_meta").expanduser().resolve()
    else:
        rsna_root = _resolve_rsna_root(ns.rsna_root)
    out_base = Path(ns.out_base).expanduser() if str(ns.out_base).strip() else Path(
        f"results/rsna_target058_shortest_{_timestamp()}"
    )
    if not out_base.is_absolute():
        out_base = (root / out_base).resolve()

    cfg = LaunchConfig(
        rsna_root=rsna_root,
        preprocessed_root=preprocessed_root,
        device=str(ns.device).strip(),
        limit_images=int(ns.limit_images),
        epochs=int(ns.epochs),
        val_frac=float(ns.val_frac),
        lr=float(ns.lr),
        preprocess=str(ns.preprocess).strip(),
        out_base=out_base,
        stack_slices=int(ns.stack_slices),
        split_by=(
            str(ns.split_by).strip()
            if ns.split_by is not None and str(ns.split_by).strip()
            else ("study" if int(ns.stack_slices) != 1 else "slice")
        ),
    )

    if ns.detach:
        _detach(list(sys.argv[1:]), cfg=cfg)
        return 0

    _run_foreground(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
