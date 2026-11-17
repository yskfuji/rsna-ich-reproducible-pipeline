from __future__ import annotations

import os
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

RSNA_LABELS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]


class ModelRef(BaseModel):
    name: str = "ich_classification__rsna_classifier"
    alias: str | None = "champion"
    version: str | None = None


class InferRequest(BaseModel):
    model_ref: ModelRef = Field(default_factory=ModelRef)
    inputs: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


def _normalize_probabilities(raw: dict[str, Any]) -> dict[str, float]:
    probs = {
        "epidural": float(raw.get("epidural", 0.02)),
        "intraparenchymal": float(raw.get("intraparenchymal", 0.08)),
        "intraventricular": float(raw.get("intraventricular", 0.04)),
        "subarachnoid": float(raw.get("subarachnoid", 0.12)),
        "subdural": float(raw.get("subdural", 0.31)),
    }
    any_input = float(raw.get("any", 0.81))
    probs["any"] = max(any_input, max(probs.values(), default=0.0))
    return probs


def create_app() -> FastAPI:
    app = FastAPI(title="rsna-ich-api-demo", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "task": "ich_classification",
            "model_family": "rsna_classifier",
        }

    @app.get("/v1/models/{alias_or_version}")
    def model_metadata(alias_or_version: str) -> dict[str, Any]:
        return {
            "task": "ich_classification",
            "model_family": "rsna_classifier",
            "alias_or_version": alias_or_version,
            "labels": RSNA_LABELS,
            "supported_architectures": ["resnet18", "efficientnet_b0", "convnext_tiny"],
            "supported_input_modes": ["2d", "2.5d"],
            "mlflow_ready": True,
            "api_ready": True,
            "notes": [
                "This public endpoint is a demo-oriented schema for portfolio review.",
                "Production model loading should resolve a concrete checkpoint or MLflow model URI.",
            ],
        }

    @app.post("/v1/infer/ich_classification")
    def infer_ich(payload: InferRequest) -> dict[str, Any]:
        started = time.perf_counter()
        probabilities = _normalize_probabilities(payload.inputs.get("probabilities", {}))
        uncertainty = float(payload.inputs.get("uncertainty", 0.12))
        return {
            "task": "ich_classification",
            "model": {
                "name": payload.model_ref.name,
                "alias": payload.model_ref.alias,
                "version": payload.model_ref.version,
            },
            "result": {
                "probabilities": probabilities,
                "uncertainty": {"score": uncertainty},
                "labels": RSNA_LABELS,
            },
            "meta": {
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "demo_mode": bool(payload.options.get("demo", True)),
                "mlflow_compatible_model_name": "ich_classification__rsna_classifier",
            },
        }

    return app


app = create_app()


def main() -> None:
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()