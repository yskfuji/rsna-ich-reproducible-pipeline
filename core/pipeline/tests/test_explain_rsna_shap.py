from __future__ import annotations

import importlib.util
import json
import sqlite3
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("shap")

PIPELINE = Path(__file__).resolve().parents[1]
SIZE = 32
N_IMAGES = 12


def _load_tool():
    spec = importlib.util.spec_from_file_location("explain_rsna_shap", PIPELINE / "tools" / "explain_rsna_shap.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _blob(x: np.ndarray) -> bytes:
    c, h, w = x.shape
    header = struct.pack("<4sBBBBHH", b"RSNP", 1, 3, 1, c, h, w)
    return header + zlib.compress(x.astype(np.float32).tobytes())


def _write_synthetic_run(root: Path, tool, *, arch: str = "resnet18") -> Path:
    """Synthetic labels, preprocessed tensors and a random checkpoint; no real images."""
    rng = np.random.default_rng(0)
    classes = list(tool.RSNA_CLASSES)
    rows = ["ID,Label"]
    db = sqlite3.connect(root / "train.sqlite")
    db.execute("CREATE TABLE tensors (key TEXT PRIMARY KEY, blob BLOB)")
    for i in range(N_IMAGES):
        image_id = f"ID_synth{i:03d}"
        positive = i % 2 == 0
        for cls in classes:
            label = int(positive and cls in ("subdural", "any"))
            rows.append(f"{image_id}_{cls},{label}")
        x = rng.random((3, SIZE, SIZE), dtype=np.float32)
        db.execute("INSERT INTO tensors VALUES (?, ?)", (image_id, _blob(x)))
    db.commit()
    db.close()
    (root / "stage_2_train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    run = root / "run"
    run.mkdir()
    meta = {"arch": arch, "pretrained": False, "image_size": SIZE, "stack_slices": 1}
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    torch.manual_seed(0)
    model = tool._build_model(arch="resnet18", in_channels=3, pretrained=False, first_conv_init="repeat")
    torch.save(model.state_dict(), run / "best.pt")
    return run


def _argv(root: Path, run: Path) -> list[str]:
    return [
        "--ckpt", str(run / "best.pt"),
        "--rsna-root", str(root),
        "--preprocessed-root", str(root),
        "--out-dir", str(root / "out"),
        "--device", "cpu",
        "--target", "any",
        "--max-samples", "1",
        "--candidate-pool", str(N_IMAGES),
        "--background-size", "2",
        "--shap-nsamples", "4",
        "--select-positive-only",
    ]


def test_explain_writes_shap_arrays_for_synthetic_run(tmp_path: Path) -> None:
    tool = _load_tool()
    run = _write_synthetic_run(tmp_path, tool)

    assert tool.main(_argv(tmp_path, run)) == 0

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state_dict"]["missing_keys"] == []
    (sample,) = manifest["samples"]
    assert sample["target"] == "any"
    assert sample["label"][-1] == 1.0
    arrays = np.load(sample["files"]["arrays"])
    assert arrays["shap"].shape == (3, SIZE, SIZE)
    assert np.isfinite(arrays["shap"]).all()
    assert float(np.abs(arrays["shap"]).sum()) > 0.0
    for key in ("overlay_signed", "overlay_abs", "window_contrib"):
        assert Path(sample["files"][key]).is_file()


def test_explain_refuses_checkpoint_that_does_not_match_arch(tmp_path: Path) -> None:
    tool = _load_tool()
    run = _write_synthetic_run(tmp_path, tool, arch="efficientnet_b0")

    with pytest.raises(RuntimeError, match="missing keys"):
        tool.main(_argv(tmp_path, run))
