"""Wait for Stage2 training to finish then run conditional-cascade evaluation (val+test).

Why:
- 10-epoch training can run unattended.
- Once best.pt exists (and optionally epoch>=expected), we automatically generate
  probmaps and run evaluation using tools/run_conditional_cascade_eval.sh.

Example:
  /opt/anaconda3/envs/medseg_unet/bin/python tools/wait_train_and_cond_cascade_eval.py \
    --exp-name medseg_3d_unet_e10_... \
    --expected-epochs 10
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import time


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]), help="Repo root (ToReBrain-pipeline)")
    ap.add_argument("--python", default="/opt/anaconda3/envs/medseg_unet/bin/python", help="Python executable for scripts")
    ap.add_argument("--exp-name", required=True, help="Experiment name under runs/3d_unet")
    ap.add_argument("--runs-root", default="runs/3d_unet")
    ap.add_argument("--expected-epochs", type=int, default=0)

    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--timeout-min", type=float, default=0.0)

    ap.add_argument("--fusion", default="residual", choices=["max", "residual"])
    ap.add_argument("--stage1-logit-eps", default="1e-4")
    ap.add_argument("--resample-max-zoom-mm", default="2.0")

    ap.add_argument("--patch-size", default="56,56,24")
    ap.add_argument("--overlap", default="0.5")
    ap.add_argument("--normalize", default="nonzero_zscore")
    ap.add_argument("--tta", default="none")

    ap.add_argument(
        "--out-dir",
        default="",
        help="If set, write results under this directory; otherwise auto under results/diag/cond_cascade_<exp>_<stamp>",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (repo / runs_root).resolve()

    exp_dir = runs_root / args.exp_name
    best_pt = exp_dir / "best.pt"
    last_meta = exp_dir / "val_threshold_last.json"

    t0 = time.time()
    while True:
        ok = False
        if best_pt.exists() and best_pt.stat().st_size > 0:
            if args.expected_epochs and args.expected_epochs > 0:
                if last_meta.exists() and last_meta.stat().st_size > 0:
                    try:
                        meta = json.loads(last_meta.read_text())
                        ep = int(meta.get("epoch") or 0)
                    except Exception:
                        ep = 0
                    ok = ep >= int(args.expected_epochs)
            else:
                ok = True

        if ok:
            break
        if args.timeout_min and (time.time() - t0) > float(args.timeout_min) * 60.0:
            raise TimeoutError(f"Timed out waiting for {best_pt}")
        time.sleep(float(args.poll_sec))

    stamp = _timestamp()
    out_dir = Path(args.out_dir) if args.out_dir else repo / "results" / "diag" / f"cond_cascade_{args.exp_name}_{stamp}"
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["PY"] = str(args.python)
    env["OUT"] = str(out_dir)
    env["MODEL"] = str(best_pt)
    env["FUSION"] = str(args.fusion)
    env["STAGE1_LOGIT_EPS"] = str(args.stage1_logit_eps)
    env["RESAMPLE_MAX_ZOOM_MM"] = str(args.resample_max_zoom_mm)
    env["PATCH_SIZE"] = str(args.patch_size)
    env["OVERLAP"] = str(args.overlap)
    env["NORM"] = str(args.normalize)
    env["TTA"] = str(args.tta)

    script = repo / "tools" / "run_conditional_cascade_eval.sh"
    cmd = ["bash", str(script), "all"]
    log_path = out_dir / "wait_train_and_cond_cascade_eval.log"
    with log_path.open("wb") as f:
        f.write(("[cmd] " + " ".join(cmd) + "\n").encode("utf-8"))
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"conditional cascade eval failed rc={rc} (see {log_path})")

    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
