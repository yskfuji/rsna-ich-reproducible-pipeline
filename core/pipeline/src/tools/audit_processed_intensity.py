"""Audit processed multi-channel volumes for intensity collapse.

This is meant to catch bugs like channel-mixed normalization that can silently
collapse small-range modalities (e.g., ADC) when paired with high-range ones (e.g., DWI).

Example:
  python -m src.tools.audit_processed_intensity \
    --root data/processed/my_dataset \
    --glob 'sub-*.nii.gz' \
    --std-thr 1e-4 \
    --out-json results/intensity_audit.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer


app = typer.Typer(add_completion=False)


@dataclass
class ChannelStats:
    min: float
    max: float
    mean: float
    std: float
    zero_frac: float
    p01: float
    p50: float
    p99: float


def _stats(arr: np.ndarray) -> ChannelStats:
    arr = arr.astype(np.float32)
    flat = arr.reshape(-1)
    zf = float((flat == 0).mean())
    p01, p50, p99 = [float(x) for x in np.percentile(flat, [1, 50, 99])]
    return ChannelStats(
        min=float(flat.min()),
        max=float(flat.max()),
        mean=float(flat.mean()),
        std=float(flat.std()),
        zero_frac=zf,
        p01=p01,
        p50=p50,
        p99=p99,
    )


@app.command()
def main(
    root: Path = typer.Option(..., help="Processed dataset root (contains images/)."),
    glob: str = typer.Option("sub-*.nii.gz", help="Glob under root/images to scan."),
    std_thr: float = typer.Option(1e-4, help="Flag channel if std < this."),
    out_json: Path | None = typer.Option(None, help="Optional: write full per-case stats JSON."),
):
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise RuntimeError("nibabel is required for auditing") from e

    img_dir = root / "images"
    paths = sorted(img_dir.glob(glob))
    if not paths:
        raise FileNotFoundError(f"No files found: {img_dir}/{glob}")

    per_case: list[dict[str, Any]] = []
    flagged: list[tuple[str, int, float, float]] = []

    # aggregate per-channel distributions
    chan_stds: dict[int, list[float]] = {}
    chan_zeros: dict[int, list[float]] = {}

    for p in paths:
        data = nib.load(str(p)).get_fdata().astype(np.float32)
        if data.ndim == 3:
            data = data[None, ...]

        rec: dict[str, Any] = {"case_id": p.stem.split(".")[0], "path": str(p), "shape": list(data.shape)}
        C = int(data.shape[0])
        rec["channels"] = []

        for c in range(C):
            st = _stats(data[c])
            rec["channels"].append(asdict(st))
            chan_stds.setdefault(c, []).append(st.std)
            chan_zeros.setdefault(c, []).append(st.zero_frac)
            if st.std < std_thr:
                flagged.append((rec["case_id"], c, st.std, st.max))

        per_case.append(rec)

    summary: dict[str, Any] = {
        "n": len(paths),
        "std_thr": std_thr,
        "flagged_count": len(flagged),
        "flagged_head": flagged[:50],
        "channels": {},
    }

    for c in sorted(chan_stds.keys()):
        stds = np.array(chan_stds[c], dtype=np.float32)
        zeros = np.array(chan_zeros[c], dtype=np.float32)
        summary["channels"][str(c)] = {
            "std_p1_p50_p99": [float(x) for x in np.percentile(stds, [1, 50, 99])],
            "zero_p1_p50_p99": [float(x) for x in np.percentile(zeros, [1, 50, 99])],
            "std_lt_thr": int(np.sum(stds < std_thr)),
        }

    print("n", summary["n"])
    print("flagged_count", summary["flagged_count"], "(std <", std_thr, ")")
    for c, info in summary["channels"].items():
        print("channel", c, "std_p1/p50/p99", info["std_p1_p50_p99"], "std_lt_thr", info["std_lt_thr"])

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            __import__("json").dumps({"summary": summary, "per_case": per_case}, indent=2, ensure_ascii=False)
        )
        print("wrote", str(out_json))


if __name__ == "__main__":
    app()
