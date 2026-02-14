from __future__ import annotations

from pathlib import Path

import yaml

import typer

from tools.data_report import main as data_report_main
from tools.loader_check import main as loader_check_main
from tools.measure_pos_patch_frac import main as sampling_main
from tools.overfit_one import main as overfit_one_main
from tools.postprocess_ablation import main as post_ablation_main
from tools.threshold_sweep import main as thr_sweep_main
from tools.inference_log import main as inference_log_main
from tools.meta_store import init_or_load_run
from tools.run_meta import meta_to_dict, write_json, write_text

app = typer.Typer(add_completion=False)


@app.command()
def main(
    # shared meta
    run_id: str = typer.Option("", help="reuse run_id (recommended to keep everything together)"),
    out_root: str = typer.Option("runs", help="runs/<run_id>/..."),
    seed: int = typer.Option(42, help="seed"),
    dataset_hash_mode: str = typer.Option("stat", help="stat|full"),
    # config for data tools
    config: str = typer.Option(..., help="train yaml (for loader_check/data_report/sampling)"),
    # model eval
    model_path: str = typer.Option(..., help="checkpoint"),
    csv_path: str = typer.Option(..., help="split csv"),
    root: str = typer.Option(..., help="processed root"),
    split_for_sweeps: str = typer.Option("val", help="split for threshold sweep"),
    split_for_infer: str = typer.Option("test", help="split for inference log + postprocess ablation"),
    patch_size: str = typer.Option("64,64,48", help="patch size"),
    overlap: float = typer.Option(0.5, help="overlap"),
    normalize: str = typer.Option("nonzero_zscore", help="normalize"),
    tta: str = typer.Option("full", help="full|flip|none"),
    # postprocess
    threshold: float = typer.Option(0.5, help="threshold for inference/postprocess ablation"),
    min_size: int = typer.Option(20, help="min_size"),
    cc_score: str = typer.Option("none", help="none|max_prob|p95_prob|mean_prob"),
    cc_score_thr: float = typer.Option(0.5, help="cc score thr"),
    # overfit
    overfit_case_id: str = typer.Option("", help="optional case_id for overfit_one"),
    # bundle runtime controls
    sampling_samples_per_split: int = typer.Option(200, help="sampling: patch samples per split (keep small; disk I/O heavy)"),
):
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(config).expanduser().resolve() if Path(config).expanduser().is_absolute() else (repo_root / config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    # Initialize a single run_id so every tool writes into the same runs/<run_id>/...
    rid_opt = run_id.strip() or None
    csv_p = Path(cfg["data"]["csv_path"]).expanduser()
    root_p = Path(cfg["data"]["root"]).expanduser()
    if not csv_p.is_absolute():
        csv_p = (repo_root / csv_p).resolve()
    if not root_p.is_absolute():
        root_p = (repo_root / root_p).resolve()

    cfg_obj = {
        "task": "deep_research_bundle",
        "config": str(cfg_path),
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "root": str(root),
        "split_for_sweeps": split_for_sweeps,
        "split_for_infer": split_for_infer,
        "patch_size": patch_size,
        "overlap": float(overlap),
        "normalize": normalize,
        "tta": tta,
        "threshold": float(threshold),
        "min_size": int(min_size),
        "cc_score": cc_score,
        "cc_score_thr": float(cc_score_thr),
        "overfit_case_id": overfit_case_id.strip() or None,
        "sampling_samples_per_split": int(sampling_samples_per_split),
    }

    meta, run_dir = init_or_load_run(
        repo_root=repo_root,
        out_root=Path(out_root),
        run_id=rid_opt,
        seed=seed,
        config_path=cfg_path,
        config_obj=cfg_obj,
        csv_path=csv_p,
        data_root=root_p,
        dataset_hash_mode=dataset_hash_mode,
    )
    rid = meta.run_id

    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    write_json(bundle_dir / "config_used.json", {"meta": meta_to_dict(meta), "config": cfg_obj})

    # (2) loader consistency
    loader_check_main(config=config, out_root=out_root, run_id=rid, seed=seed, dataset_hash_mode=dataset_hash_mode)

    # (3) data distribution report
    data_report_main(config=config, out_root=out_root, run_id=rid, seed=seed, dataset_hash_mode=dataset_hash_mode)

    # (7) sampling pos_patch_frac measurement
    sampling_main(
        config=config,
        out_root=out_root,
        run_id=rid,
        seed=seed,
        dataset_hash_mode=dataset_hash_mode,
        samples_per_split=int(sampling_samples_per_split),
        splits="train,val,test",
    )

    # (5) threshold sweep
    thr_sweep_main(
        model_path=model_path,
        csv_path=csv_path,
        root=root,
        split=split_for_sweeps,
        out_root=out_root,
        run_id=rid,
        seed=seed,
        dataset_hash_mode=dataset_hash_mode,
        patch_size=patch_size,
        overlap=overlap,
        normalize=normalize,
        tta=tta,
    )

    # (5b) threshold sweep on test with min_size=0 (no postprocess), step=0.05
    thr_sweep_main(
        model_path=model_path,
        csv_path=csv_path,
        root=root,
        split=split_for_infer,
        out_root=out_root,
        run_id=rid,
        out_subdir="threshold_sweep_test_min0",
        seed=seed,
        dataset_hash_mode=dataset_hash_mode,
        patch_size=patch_size,
        overlap=overlap,
        step=0.05,
        min_size=0,
        cc_score="none",
        normalize=normalize,
        tta=tta,
    )

    # (4) inference log
    inference_log_main(
        model_path=model_path,
        csv_path=csv_path,
        root=root,
        split=split_for_infer,
        out_root=out_root,
        run_id=rid,
        seed=seed,
        dataset_hash_mode=dataset_hash_mode,
        patch_size=patch_size,
        overlap=overlap,
        thresholds=str(threshold),
        min_size=int(min_size),
        cc_score=cc_score,
        cc_score_thr=float(cc_score_thr),
        normalize=normalize,
        tta=tta,
    )

    # (6) postprocess on/off comparison
    post_ablation_main(
        model_path=model_path,
        csv_path=csv_path,
        root=root,
        split=split_for_infer,
        out_root=out_root,
        run_id=rid,
        seed=seed,
        dataset_hash_mode=dataset_hash_mode,
        patch_size=patch_size,
        overlap=overlap,
        threshold=float(threshold),
        on_min_size=int(min_size),
        on_cc_score=cc_score,
        on_cc_score_thr=float(cc_score_thr),
        normalize=normalize,
        tta=tta,
    )

    # (1) optional: overfit one
    if overfit_case_id.strip():
        overfit_one_main(config=config, case_id=overfit_case_id.strip(), out_root=out_root, run_id=rid, seed=seed, dataset_hash_mode=dataset_hash_mode)

    # top-level pointer
    write_text(
        bundle_dir / "report.md",
        "\n".join(
            [
                f"# Deep Research Bundle ({rid})",
                "",
                "## Contents",
                "- `loader_check/report.md`",
                "- `data_report/report.md`",
                "- `sampling/report.md`",
                "- `threshold_sweep/report.md`",
                "- `threshold_sweep_test_min0/report.md`",
                "- `inference/report.md`",
                "- `postprocess_ablation/diff.md`",
                "- `overfit_one/report.md` (optional)",
            ]
        )
        + "\n",
    )


if __name__ == "__main__":
    app()
