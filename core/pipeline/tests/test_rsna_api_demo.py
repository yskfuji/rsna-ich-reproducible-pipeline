from __future__ import annotations

from src.inference.serve_rsna_ich_api import InferRequest, ModelRef, create_app


def test_rsna_api_routes_exist() -> None:
    app = create_app()
    paths = sorted(route.path for route in app.routes if hasattr(route, "path"))
    assert "/health" in paths
    assert "/v1/models/{alias_or_version}" in paths
    assert "/v1/infer/ich_classification" in paths


def test_rsna_api_infer_normalizes_any_score() -> None:
    app = create_app()
    infer_route = next(route for route in app.routes if getattr(route, "path", None) == "/v1/infer/ich_classification")
    payload = InferRequest(
        model_ref=ModelRef(name="ich_classification__rsna_classifier", alias="champion"),
        inputs={"probabilities": {"subdural": 0.4, "epidural": 0.1, "any": 0.2}},
        options={"demo": True},
    )
    response = infer_route.endpoint(payload)
    assert response["result"]["probabilities"]["subdural"] == 0.4
    assert response["result"]["probabilities"]["any"] == 0.4