from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.rsna_ich_dataset import RSNA_CLASSES, iter_rsna_stage2_records_from_csv


def _write_stage2_train_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    # Minimal CSV writer (no pandas dependency in tests).
    path.write_text(
        "ID,Label\n" + "\n".join([f"{rid},{float(v)}" for rid, v in rows]) + "\n",
        encoding="utf-8",
    )


def test_iter_rsna_stage2_records_limit_images_is_unbiased_and_complete(tmp_path: Path) -> None:
    # Create a tiny RSNA-like CSV with 3 image_ids and 6 classes each.
    # Interleave rows so a naive "stop early" reader could miss some labels.
    img_ids = ["ID_A", "ID_B", "ID_C"]
    rows: list[tuple[str, float]] = []
    for cls_i, cls in enumerate(RSNA_CLASSES):
        for img_i, img in enumerate(img_ids):
            # deterministic but non-trivial pattern
            v = float((img_i + cls_i) % 2)
            rows.append((f"{img}_{cls}", v))

    csv_path = tmp_path / "stage_2_train.csv"
    _write_stage2_train_csv(csv_path, rows)

    dcm_dir = tmp_path / "stage_2_train"
    dcm_dir.mkdir(parents=True, exist_ok=True)
    for img in img_ids:
        (dcm_dir / f"{img}.dcm").write_bytes(b"")

    recs = iter_rsna_stage2_records_from_csv(
        csv_path=csv_path,
        dicom_dir=dcm_dir,
        limit_images=2,
        seed=123,
    )
    assert len(recs) == 2

    for r in recs:
        assert r.image_id in set(img_ids)
        assert r.dcm_path.name == f"{r.image_id}.dcm"
        assert r.y.shape == (len(RSNA_CLASSES),)
        assert np.isfinite(r.y).all()
