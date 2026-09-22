"""FastAPI app: 1 Hz SSE of engine state, plus a static single-page frontend."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .engine import Engine
from .model import precision_at_base_rate

log = logging.getLogger("dashboard.server")
STATIC = Path(__file__).parent / "static"


def _seen_level(m, engine) -> str:
    """How much of what we are watching this model has already been trained on.

    "run"    -- fitted on this exact run. Its output here is not evidence of
                anything; it is scoring its own training data.
    "config" -- never saw this run, but was trained on this workload
                configuration. Better, still flattering: this is the gap
                between grouped-by-run and grouped-by-workload accuracy.
    "none"   -- never saw this configuration at all. The honest case.
    "unknown"-- the bundle does not record what it trained on.
    """
    runs = set(m.metadata.get("train_run_ids") or [])
    labels = set(m.metadata.get("train_workload_labels") or [])
    if not runs and not labels:
        return "unknown"

    replayed = engine.source.info.run_ids or []
    if replayed and any(r in runs for r in replayed):
        return "run"

    # Live NVML has no run id, so the operator's declared label is the only
    # thing that says which workload this is.
    declared = engine.events.declared_label
    if declared and declared in labels:
        return "config"
    if replayed:
        # Replayed traces carry their label in the filename-derived run label.
        for name in engine.source.info.trace_names or []:
            if name.split(" (")[0] in labels:
                return "config"
    return "none"


def _corpus_summary(m) -> dict:
    return {
        "n_windows": m.metadata.get("n_windows"),
        "n_runs": m.metadata.get("n_runs"),
        "n_configs": m.metadata.get("n_configs"),
        "runs_per_class": m.metadata.get("runs_per_class"),
        "configs_per_class": m.metadata.get("configs_per_class"),
        "created_utc": m.metadata.get("created_utc"),
        "holdout_configs": m.metadata.get("holdout_configs") or [],
    }


def model_card(engine: Engine) -> dict:
    """Everything the UI needs to state what this model is and is not."""
    m = engine.model
    op = m.binary_operating_point()

    curve = None
    if op:
        curve = [
            {"base_rate": b,
             "precision": precision_at_base_rate(op["recall"], op["fpr"], b)}
            for b in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.62, 0.80)
        ]

    return {
        "path": str(m.path),
        "task": m.task,
        "classes": m.classes,
        "window_sec": m.window_sec,
        "stride_sec": m.stride_sec,
        "feature_variant": m.feature_variant,
        "n_features": len(m.feature_cols),
        "provenance": m.provenance,
        "trained_power_w": m.trained_power_w,
        "corpus": _corpus_summary(m),
        "scores": m.scores(),
        "seen_level": _seen_level(m, engine),
        "holdout_configs": m.metadata.get("holdout_configs") or [],
        "operating_point": op,
        "precision_curve": curve,
        "provenance_warnings": engine.provenance_warnings,
        "compare": _compare_card(engine),
        "model_id": _model_id_card(engine),
    }


def _model_id_card(engine: Engine) -> dict | None:
    m = engine.model_id_model
    if m is None:
        return None
    md = m.metadata
    return {
        "path": str(m.path),
        "classes": m.classes,
        "n_features": len(m.feature_cols),
        "memory_features_removed": m.extras.get("memory_features_removed",
                                                 md.get("memory_features_removed")),
        "serving_config": md.get("serving_config"),
        "n_runs": md.get("n_runs"),
        "n_models": md.get("n_models"),
        "n_windows": md.get("n_windows"),
        "scores": md.get("scores"),
        "models": md.get("models"),
        "corpus": md.get("corpus"),
        "seen_level": _seen_level(m, engine),
        "provenance_warnings": m.compare_to_source(engine.source.info),
        "trained_power_w": m.trained_power_w,
    }


def _compare_card(engine: Engine) -> dict | None:
    c = engine.compare_model
    if c is None:
        return None
    holdout = c.metadata.get("holdout_configs") or []
    return {
        "path": str(c.path),
        "name": c.path.parent.name,
        "feature_variant": c.feature_variant,
        "n_features": len(c.feature_cols),
        "classes": c.classes,
        "corpus": _corpus_summary(c),
        "scores": c.scores(),
        "holdout_configs": holdout,
        "trained_power_w": c.trained_power_w,
        "provenance": c.provenance,
        # The comparison model is checked against this machine too. Without
        # this, a bundle from the wrong power regime could sit in the compare
        # slot looking authoritative while the banner -- which only inspects
        # the primary -- stayed green.
        "provenance_warnings": c.compare_to_source(engine.source.info),
        # How much of what we are watching this model already trained on --
        # the one fact that decides whether its numbers mean anything.
        "seen_level": _seen_level(c, engine),
    }


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="GPU workload classifier — live", docs_url=None,
                  redoc_url=None)

    @app.on_event("startup")
    async def _startup():
        engine.start()

    @app.on_event("shutdown")
    async def _shutdown():
        engine.stop()

    @app.get("/api/state")
    async def state():
        return engine.state()

    @app.get("/api/model")
    async def model():
        return model_card(engine)

    @app.get("/api/stream")
    async def stream():
        """One state frame per second. SSE because the flow is one-directional."""
        async def gen():
            # The model card is static for the process; send it once up front so
            # the page can render its banner before the first tick.
            yield f"event: model\ndata: {json.dumps(model_card(engine))}\n\n"
            while True:
                payload = json.dumps(engine.state(), default=float)
                yield f"event: state\ndata: {payload}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/mark")
    async def mark(payload: dict):
        """Declare what the GPU is actually doing, so the panels have a truth.

        Live NVML carries no ground truth. Without a mark, the transition and
        false-positive panels stay blank rather than guessing.
        """
        label = payload.get("label")
        if label is not None and not isinstance(label, str):
            raise HTTPException(400, "label must be a string or null")
        engine.events.declare(label or None, t=engine.stream_now)
        return {"declared_label": engine.events.declared_label,
                "declared_class": engine.events.declared_class}

    @app.post("/api/reset")
    async def reset():
        engine.reset()
        return {"ok": True}

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
