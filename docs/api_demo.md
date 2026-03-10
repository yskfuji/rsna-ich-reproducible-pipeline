# RSNA API demo

This repository now includes a lightweight FastAPI demo surface for the public RSNA package.

## Purpose

- show how the RSNA task can be exposed through a typed HTTP schema
- make the MLflow/API direction visible in the public portfolio bundle
- provide a small demo endpoint without bundling protected medical data or production checkpoints

## Entry point

- [core/pipeline/serve_rsna_ich_api.py](../core/pipeline/serve_rsna_ich_api.py)

## Endpoints

- `GET /health`
- `GET /v1/models/{alias_or_version}`
- `POST /v1/infer/ich_classification`

## What this demo does

- returns typed metadata for the RSNA task and model family
- exposes a mock-but-structured inference response for `ich_classification`
- normalizes the `any` probability so it is at least the maximum subtype probability

## What this demo does not do

- it does not load protected medical data
- it does not claim production deployment readiness
- it does not yet resolve a real MLflow model URI or checkpoint alias

## Local run

```bash
cd core/pipeline
python serve_rsna_ich_api.py
```

Open `http://127.0.0.1:8000/docs` to inspect the schema.

## Intended next step

The next public-facing extension is to replace the demo response with a real model-loading path backed by a checkpoint or MLflow model reference.