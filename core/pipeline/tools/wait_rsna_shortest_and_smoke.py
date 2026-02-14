from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Optional
from datetime import datetime
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _exists_best(run_dir: Path) -> bool:
    return (run_dir / "best_wlogloss.pt").exists() or (run_dir / "best.pt").exists()


def _resolve_rsna_root(maybe_rsna_root: Optional[str]) -> Path:
    value = (maybe_rsna_root or os.environ.get("RSNA_ROOT") or "").strip()
    if not value:
        raise SystemExit(
            "Missing RSNA root. Provide --rsna-root or set env RSNA_ROOT to the dataset root "
            "(contains stage_2_train.csv and stage_2_train/; stage_2_test/ and stage_2_sample_submission.csv)."
        )
    return Path(value).expanduser().resolve()


def _resolve_preprocessed_root(maybe: Optional[str]) -> Path | None:
    v = (maybe or os.environ.get("RSNA_PREPROCESSED_ROOT") or "").strip()
    if not v:
        return None
    return Path(v).expanduser().resolve()


def _wait_for_runs(out_base: Path, *, epochs: int, stack_slices: int, poll_sec: int, log_path: Path) -> tuple[Path, Path]:
    run_tag = "2d" if int(stack_slices) == 1 else f"25d_stack{int(stack_slices)}"
    eff = out_base / f"effb0_{run_tag}_img384_slicesplit_e{epochs}"
    cnx = out_base / f"convnext_tiny_{run_tag}_img384_slicesplit_e{epochs}"

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_ts()}] waiting for checkpoints...\n")
        log.write(f"  eff={eff}\n  cnx={cnx}\n")
        log.flush()

        while True:
            ok1 = eff.exists() and _exists_best(eff)
            ok2 = cnx.exists() and _exists_best(cnx)
            if ok1 and ok2:
                log.write(f"[{_ts()}] found checkpoints\n")
                log.flush()
                return eff, cnx
            log.write(f"[{_ts()}] not ready: eff={ok1} cnx={ok2}\n")
            log.flush()
            time.sleep(poll_sec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rsna-root", default=None, help="RSNA dataset root. If omitted, uses env RSNA_ROOT.")
    p.add_argument(
        "--preprocessed-root",
        default=None,
        help="If set, run smoke inference from <preprocessed_root>/test.sqlite (DICOM not required). "
        "If --rsna-root is omitted, defaults to <preprocessed_root>/rsna_meta for CSVs.",
    )
    p.add_argument("--out-base", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument(
        "--stack-slices",
        type=int,
        default=1,
        help="Must match the value used in launch script (1=2D, 3=2.5D, etc).",
    )
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--max-test-images",
        type=int,
        default=200,
        help="If >0, limits inference to N test images via MAX_TEST_IMAGES. If 0, runs full inference.",
    )
    p.add_argument("--poll-sec", type=int, default=60)
    p.add_argument("--out-csv", default="submission_ensemble_smoke.csv")

    ns = p.parse_args(argv)

    root = _project_root()
    preprocessed_root = _resolve_preprocessed_root(ns.preprocessed_root)
    if ns.rsna_root is None and preprocessed_root is not None:
        rsna_root = (preprocessed_root / "rsna_meta").expanduser().resolve()
    else:
        rsna_root = _resolve_rsna_root(ns.rsna_root)
    out_base = Path(ns.out_base).expanduser().resolve()
    out_csv = Path(ns.out_csv).expanduser().resolve() if Path(ns.out_csv).is_absolute() else (root / ns.out_csv).resolve()

    log_path = out_base / "smoke_wait.log"

    eff_dir, cnx_dir = _wait_for_runs(
        out_base,
        epochs=int(ns.epochs),
        stack_slices=int(ns.stack_slices),
        poll_sec=int(ns.poll_sec),
        log_path=log_path,
    )

    # Run smoke ensemble via existing zsh wrapper.
    # IMPORTANT: do not build a shell command string here; RSNA_ROOT may include spaces and apostrophes.
    max_test_images = int(ns.max_test_images)
    env = dict(os.environ)
    env["TORCH_DEVICE"] = str(ns.device)
    if max_test_images > 0:
        env["MAX_TEST_IMAGES"] = str(max_test_images)
    if preprocessed_root is not None:
        env["RSNA_PREPROCESSED_ROOT"] = str(preprocessed_root)

    script = root / "scripts" / "make_rsna_submission_from_two_bests.zsh"
    cmd = [
        "zsh",
        str(script),
        str(rsna_root),
        str(eff_dir),
        str(cnx_dir),
        str(out_csv),
        str(ns.device),
    ]

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_ts()}] running smoke ensemble\n")
        log.write(f"[{_ts()}] cmd={' '.join(cmd)}\n")
        log.flush()
        subprocess.check_call(cmd, cwd=str(root), env=env, stdout=log, stderr=log)
        log.write(f"[{_ts()}] done: out_csv={out_csv}\n")
        log.flush()

    print(f"[done] {out_csv}")
    print(f"[log] {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
