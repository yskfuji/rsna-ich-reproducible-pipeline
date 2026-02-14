"""Run multiple training configs sequentially with robust logging.

Motivation:
- VS Code terminal sessions may disconnect.
- Training is long; we want a single background process to run a queue.

Usage:
  /opt/anaconda3/envs/medseg_unet/bin/python tools/run_train_queue.py \
    --python /opt/anaconda3/envs/medseg_unet/bin/python \
    --repo /Users/yusukefujinami/ToReBrain/ToReBrain-pipeline \
    --configs configs/a.yaml configs/b.yaml

Notes:
- Sets PYTHONPATH to the repo directory (so `-m src.training.train_3d_unet` works).
- Writes one log file per config under results/diag/train_queue_logs/.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]), help="Repo root (ToReBrain-pipeline)")
    ap.add_argument("--configs", nargs="+", required=True, help="List of YAML config paths")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    py = str(Path(args.python).resolve())

    log_root = repo / "results" / "diag" / "train_queue_logs" / f"queue_{_timestamp()}"
    log_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    print(f"[queue] repo={repo}")
    print(f"[queue] python={py}")
    print(f"[queue] logs={log_root}")
    print(f"[queue] n_configs={len(args.configs)}")

    for idx, cfg in enumerate(args.configs, start=1):
        cfg_path = Path(cfg)
        if not cfg_path.is_absolute():
            cfg_path = (repo / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))

        log_path = log_root / f"{idx:02d}_{cfg_path.stem}.log"
        meta_path = log_root / f"{idx:02d}_{cfg_path.stem}.meta.txt"

        print(f"[queue] ({idx}/{len(args.configs)}) start {cfg_path.name}")
        meta_path.write_text(
            "\n".join(
                [
                    f"timestamp={_timestamp()}",
                    f"python={py}",
                    f"repo={repo}",
                    f"config={cfg_path}",
                ]
            )
            + "\n"
        )

        cmd = [py, "-m", "src.training.train_3d_unet", "--config", str(cfg_path)]
        with log_path.open("wb") as f:
            f.write(("[cmd] " + " ".join(cmd) + "\n").encode("utf-8"))
            f.flush()
            p = subprocess.Popen(
                cmd,
                cwd=str(repo),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            rc = p.wait()
            f.write((f"\n[exit] code={rc}\n").encode("utf-8"))

        if rc != 0:
            print(f"[queue] FAILED config={cfg_path} rc={rc} (see {log_path})")
            return rc

        print(f"[queue] done {cfg_path.name} (log={log_path.name})")

    print("[queue] ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
