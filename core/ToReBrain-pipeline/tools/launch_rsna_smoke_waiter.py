from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_preprocessed_root(maybe: str | None) -> Path | None:
    v = (maybe or os.environ.get("RSNA_PREPROCESSED_ROOT") or "").strip()
    if not v:
        return None
    return Path(v).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preprocessed-root",
        default=None,
        type=str,
        help="If set, passes RSNA_PREPROCESSED_ROOT to the waiter so inference uses SQLite (DICOM not required).",
    )
    p.add_argument("--out-base", required=True, type=str)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--stack-slices", type=int, default=1)
    p.add_argument("--device", default="mps", type=str)
    p.add_argument("--max-test-images", type=int, default=200)
    p.add_argument("--poll-sec", type=int, default=300)
    p.add_argument(
        "--out-csv",
        default=None,
        type=str,
        help="Output CSV path. Default: <out-base>/submission_smoke.csv",
    )

    ns = p.parse_args(argv)

    root = _project_root()
    out_base = Path(ns.out_base).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    out_csv = (
        Path(ns.out_csv).expanduser().resolve()
        if ns.out_csv is not None and str(ns.out_csv).strip()
        else (out_base / "submission_smoke.csv")
    )

    waiter = root / "tools" / "wait_rsna_shortest_and_smoke.py"
    if not waiter.exists():
        raise FileNotFoundError(f"Missing: {waiter}")

    env = dict(os.environ)
    env["TORCH_DEVICE"] = str(ns.device)
    env.setdefault("PYTHONUNBUFFERED", "1")

    preprocessed_root = _resolve_preprocessed_root(ns.preprocessed_root)
    if preprocessed_root is not None:
        env["RSNA_PREPROCESSED_ROOT"] = str(preprocessed_root)

    cmd = [
        sys.executable,
        str(waiter),
        "--out-base",
        str(out_base),
        "--epochs",
        str(int(ns.epochs)),
        "--stack-slices",
        str(int(ns.stack_slices)),
        "--device",
        str(ns.device),
        "--max-test-images",
        str(int(ns.max_test_images)),
        "--poll-sec",
        str(int(ns.poll_sec)),
        "--out-csv",
        str(out_csv),
    ]

    log_path = out_base / "smoke_wait.launcher.log"
    log_f = log_path.open("a", encoding="utf-8")
    p_wait = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,
        close_fds=True,
    )

    print(f"[launched waiter] pid={p_wait.pid}")
    print(f"out_base={out_base}")
    print(f"out_csv={out_csv}")
    print(f"log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
