from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_NAME = "github_public_rsna"
TASK_TYPE = "medical_image_classification"
MODEL_FAMILY = "cnn2d_classifier"
TRACKING_SCHEMA = "public_portfolio_v1"
DEFAULT_CHECKPOINT_CANDIDATES = ["best_wlogloss.pt", "best.pt", "best_auc.pt", "last.pt", "last_state.pt"]
DEFAULT_METADATA_FILES = ["meta.json"]
DEFAULT_TRACE_FILES = ["log.jsonl"]
COMPARISON_OPERATORS = {
    ">=": lambda actual, threshold: actual >= threshold,
    "<=": lambda actual, threshold: actual <= threshold,
    ">": lambda actual, threshold: actual > threshold,
    "<": lambda actual, threshold: actual < threshold,
    "==": lambda actual, threshold: actual == threshold,
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_last(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_if_exists(src: Path, dst: Path, bundle_root: Path) -> dict[str, Any] | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": str(dst.relative_to(bundle_root)),
        "source_path": str(src),
        "size_bytes": int(dst.stat().st_size),
        "sha256": _sha256_file(dst),
    }


def _coerce_metric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_promotion_rule(rule_text: str) -> tuple[str, str, float]:
    for operator in (">=", "<=", "==", ">", "<"):
        if operator in rule_text:
            metric_name, threshold_text = rule_text.split(operator, 1)
            metric_name = metric_name.strip()
            threshold_text = threshold_text.strip()
            if not metric_name or not threshold_text:
                break
            return metric_name, operator, float(threshold_text)
    raise ValueError(f"invalid promotion rule: {rule_text}")


def _evaluate_promotion_rules(latest_metrics: dict[str, Any], rule_texts: list[str]) -> dict[str, Any]:
    if not rule_texts:
        return {"status": "not_requested", "passed": None, "rules": []}

    evaluations: list[dict[str, Any]] = []
    passed = True
    for rule_text in rule_texts:
        metric_name, operator, threshold = _parse_promotion_rule(rule_text)
        raw_value = latest_metrics.get(metric_name)
        actual_value = _coerce_metric_value(raw_value)
        if actual_value is None:
            rule_passed = False
            reason = "metric missing or non-numeric"
        else:
            rule_passed = COMPARISON_OPERATORS[operator](actual_value, threshold)
            reason = None
        evaluations.append(
            {
                "rule": rule_text,
                "metric_name": metric_name,
                "operator": operator,
                "threshold": threshold,
                "actual_value": actual_value,
                "passed": rule_passed,
                "reason": reason,
            }
        )
        passed = passed and rule_passed

    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "rules": evaluations,
    }


def _safe_mlflow_register(
    bundle_dir: Path,
    registration_path: Path,
    registration: dict[str, Any],
    *,
    tracking_uri: str | None,
    experiment_name: str,
    registered_model_name: str | None,
    registered_model_alias: str | None,
    promote_alias: str | None,
    reject_alias: str | None,
    strict: bool,
) -> dict[str, Any] | None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        if strict:
            raise
        result = {"status": "skipped", "warning": f"mlflow import failed: {exc}"}
        registration["mlflow_registration"] = result
        registration_path.write_text(json.dumps(registration, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    result: dict[str, Any] = {"status": "logged"}
    with mlflow.start_run(run_name=f"register-{registration['model_name']}-{registration['version_label']}") as run:
        run_id = run.info.run_id
        promotion_status = registration["promotion_evaluation"]["status"]
        result["registration_run_id"] = run_id
        result["promotion_status"] = promotion_status
        mlflow.set_tags(
            {
                "repo_name": REPO_NAME,
                "task_type": TASK_TYPE,
                "model_family": MODEL_FAMILY,
                "tracking_schema": TRACKING_SCHEMA,
                "promotion_stage": "registration_bundle",
                "promotion_status": promotion_status,
                "model_name": registration["model_name"],
                "version_label": registration["version_label"],
            }
        )
        mlflow.log_params(
            {
                "source_run_dir": registration["source_run_dir"],
                "primary_checkpoint": registration["primary_checkpoint"],
                "bundle_dir": str(bundle_dir),
                "promotion_rule_count": len(registration["promotion_evaluation"].get("rules", [])),
            }
        )
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")

        if registered_model_name:
            client = MlflowClient()
            source_uri = f"runs:/{run_id}/bundle"
            result["registered_model_name"] = registered_model_name
            result["registered_model_source_uri"] = source_uri
            try:
                client.create_registered_model(registered_model_name)
            except Exception:
                pass
            try:
                model_version = client.create_model_version(
                    name=registered_model_name,
                    source=source_uri,
                    run_id=run_id,
                    description=f"Registration bundle for {registration['model_name']}:{registration['version_label']}",
                )
                result["registered_model_version"] = str(model_version.version)
                alias_to_apply = registered_model_alias
                if registration["promotion_evaluation"]["passed"] is True and promote_alias:
                    alias_to_apply = promote_alias
                elif registration["promotion_evaluation"]["passed"] is False and reject_alias:
                    alias_to_apply = reject_alias
                if alias_to_apply:
                    client.set_registered_model_alias(registered_model_name, alias_to_apply, model_version.version)
                    result["registered_model_alias"] = alias_to_apply
            except Exception as exc:
                if strict:
                    raise
                result["warning"] = f"model version creation failed: {exc}"

        registration["mlflow_registration"] = result
        registration_path.write_text(json.dumps(registration, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(registration_path), artifact_path="bundle")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a registry-ready model bundle from a training run.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing meta/log/checkpoints")
    parser.add_argument("--model-name", default="", help="Public model name for the registration bundle")
    parser.add_argument("--version-label", default="", help="Version label for the registration bundle")
    parser.add_argument("--bundle-root", default="artifacts/registered_models", help="Base directory for output bundles")
    parser.add_argument("--checkpoint", default="", help="Primary checkpoint file name inside the run dir")
    parser.add_argument("--selection-reason", default="", help="Human-readable reason for promoting this run")
    parser.add_argument("--bundle-note", default="", help="Optional note stored in registration.json")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing bundle directory")
    parser.add_argument("--mlflow-register", action="store_true", help="Log the registration bundle to MLflow")
    parser.add_argument("--mlflow-tracking-uri", default="", help="Optional MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", default="model-registration", help="MLflow experiment name for registration runs")
    parser.add_argument("--registered-model-name", default="", help="Optional MLflow registered model name")
    parser.add_argument("--registered-model-alias", default="", help="Optional MLflow registered model alias")
    parser.add_argument("--promotion-rule", action="append", default=[], help="Promotion rule such as val_dice>=0.75 or val_logloss_weighted<=0.055")
    parser.add_argument("--promote-alias", default="", help="Alias to assign when all promotion rules pass")
    parser.add_argument("--reject-alias", default="", help="Alias to assign when promotion rules fail")
    parser.add_argument("--strict-mlflow", action="store_true", help="Fail if MLflow registration steps fail")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    meta = _read_json(run_dir / "meta.json") or {}
    last_log = _read_jsonl_last(run_dir / "log.jsonl") or {}
    model_name = args.model_name.strip() or str(meta.get("arch") or run_dir.name)
    version_label = args.version_label.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    primary_checkpoint_name = args.checkpoint.strip()
    if not primary_checkpoint_name:
        for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
            if (run_dir / candidate).exists():
                primary_checkpoint_name = candidate
                break
    if not primary_checkpoint_name:
        raise FileNotFoundError(f"no checkpoint found under {run_dir}")

    bundle_dir = Path(args.bundle_root).expanduser().resolve() / model_name / version_label
    if bundle_dir.exists():
        if not args.force:
            raise FileExistsError(f"bundle already exists: {bundle_dir}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[dict[str, Any]] = []
    for name in DEFAULT_METADATA_FILES:
        item = _copy_if_exists(run_dir / name, bundle_dir / "run_metadata" / name, bundle_dir)
        if item is not None:
            inventory.append(item)
    for name in DEFAULT_TRACE_FILES:
        item = _copy_if_exists(run_dir / name, bundle_dir / "training_trace" / name, bundle_dir)
        if item is not None:
            inventory.append(item)
    for name in DEFAULT_CHECKPOINT_CANDIDATES:
        item = _copy_if_exists(run_dir / name, bundle_dir / "checkpoints" / name, bundle_dir)
        if item is not None:
            inventory.append(item)

    primary_bundle_checkpoint = bundle_dir / "checkpoints" / primary_checkpoint_name
    if not primary_bundle_checkpoint.exists():
        raise FileNotFoundError(f"primary checkpoint missing from bundle: {primary_bundle_checkpoint}")

    promotion_evaluation = _evaluate_promotion_rules(last_log, list(args.promotion_rule))

    registration = {
        "schema_version": "model_registration_bundle.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_name": REPO_NAME,
        "task_type": TASK_TYPE,
        "model_family": MODEL_FAMILY,
        "tracking_schema": TRACKING_SCHEMA,
        "model_name": model_name,
        "version_label": version_label,
        "source_run_dir": str(run_dir),
        "primary_checkpoint": f"checkpoints/{primary_checkpoint_name}",
        "selection_reason": args.selection_reason.strip() or "manual promotion",
        "bundle_note": args.bundle_note.strip() or None,
        "source_meta": meta,
        "latest_metrics": last_log,
        "promotion_evaluation": promotion_evaluation,
        "artifact_inventory": inventory,
    }
    registration_path = bundle_dir / "registration.json"
    registration_path.write_text(json.dumps(registration, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.mlflow_register:
        _safe_mlflow_register(
            bundle_dir,
            registration_path,
            registration,
            tracking_uri=args.mlflow_tracking_uri.strip() or None,
            experiment_name=args.mlflow_experiment.strip(),
            registered_model_name=args.registered_model_name.strip() or None,
            registered_model_alias=args.registered_model_alias.strip() or None,
            promote_alias=args.promote_alias.strip() or None,
            reject_alias=args.reject_alias.strip() or None,
            strict=bool(args.strict_mlflow),
        )

    print(
        json.dumps(
            {
                "bundle_dir": str(bundle_dir),
                "registration_path": str(registration_path),
                "promotion_status": promotion_evaluation["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())