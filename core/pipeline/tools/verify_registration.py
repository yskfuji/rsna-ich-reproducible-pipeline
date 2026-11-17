from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run register_model.py against a representative run and verify registration.json plus MLflow alias state."
    )
    parser.add_argument("--run-dir", required=True, help="Training run directory to verify")
    parser.add_argument("--model-name", required=True, help="Model name passed to register_model.py")
    parser.add_argument("--version-label", default="", help="Version label for the verification bundle")
    parser.add_argument("--checkpoint", default="", help="Checkpoint file name inside the run dir")
    parser.add_argument("--selection-reason", default="verification run", help="Reason stored in registration.json")
    parser.add_argument("--promotion-rule", action="append", default=[], help="Promotion rule such as val_dice>=0.75")
    parser.add_argument("--verification-root", default="artifacts/verification", help="Directory for local verification outputs")
    parser.add_argument("--tracking-uri", default="", help="Optional MLflow tracking URI override")
    parser.add_argument("--mlflow-experiment", default="model-registration-verify", help="MLflow experiment name")
    parser.add_argument("--registered-model-name", default="", help="Registered model name used for alias verification")
    parser.add_argument("--promote-alias", default="candidate", help="Alias expected when promotion passes")
    parser.add_argument("--reject-alias", default="needs-review", help="Alias expected when promotion fails")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing verification bundle")
    return parser


def _build_tracking_uri(verification_root: Path, tracking_uri: str) -> str:
    if tracking_uri.strip():
        return tracking_uri.strip()
    return f"sqlite:///{(verification_root / 'mlflow.db').resolve()}"


def _run_register_model(args: argparse.Namespace, verification_root: Path, tracking_uri: str) -> Path:
    version_label = args.version_label.strip() or datetime.now(timezone.utc).strftime("verify-%Y%m%dT%H%M%SZ")
    bundle_root = (verification_root / "registered_models").resolve()
    register_script = Path(__file__).with_name("register_model.py")
    command = [
        sys.executable,
        str(register_script),
        "--run-dir",
        str(Path(args.run_dir).expanduser().resolve()),
        "--model-name",
        args.model_name.strip(),
        "--version-label",
        version_label,
        "--bundle-root",
        str(bundle_root),
        "--selection-reason",
        args.selection_reason.strip(),
        "--mlflow-register",
        "--mlflow-tracking-uri",
        tracking_uri,
        "--mlflow-experiment",
        args.mlflow_experiment.strip(),
        "--registered-model-name",
        args.registered_model_name.strip() or f"{args.model_name.strip()}-verify",
        "--promote-alias",
        args.promote_alias.strip(),
        "--reject-alias",
        args.reject_alias.strip(),
    ]
    if args.checkpoint.strip():
        command.extend(["--checkpoint", args.checkpoint.strip()])
    if args.force:
        command.append("--force")
    for rule in args.promotion_rule:
        command.extend(["--promotion-rule", rule])
    subprocess.run(command, check=True)
    return bundle_root / args.model_name.strip() / version_label / "registration.json"


def main() -> int:
    args = build_parser().parse_args()
    verification_root = Path(args.verification_root).expanduser().resolve()
    verification_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = _build_tracking_uri(verification_root, args.tracking_uri)
    registration_path = _run_register_model(args, verification_root, tracking_uri)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))

    expected_alias = None
    if registration["promotion_evaluation"]["passed"] is True:
        expected_alias = args.promote_alias.strip() or None
    elif registration["promotion_evaluation"]["passed"] is False:
        expected_alias = args.reject_alias.strip() or None

    alias_result: dict[str, Any] | None = None
    if expected_alias:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        registered_model_name = args.registered_model_name.strip() or f"{args.model_name.strip()}-verify"
        version = client.get_model_version_by_alias(registered_model_name, expected_alias)
        alias_result = {
            "registered_model_name": registered_model_name,
            "alias": expected_alias,
            "version": str(version.version),
        }

    print(
        json.dumps(
            {
                "registration_path": str(registration_path),
                "promotion_status": registration["promotion_evaluation"]["status"],
                "expected_alias": expected_alias,
                "alias_verification": alias_result,
                "tracking_uri": tracking_uri,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())