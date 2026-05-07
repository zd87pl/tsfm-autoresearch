"""M10 GCP Scale-Out: FastAPI forecast service for Cloud Run.

Stateless worker that loads TimesFM once at startup and serves
autoresearch forecast requests over HTTP.

Designed for:
  - Cloud Run (serverless, auto-scale)
  - GKE (Kubernetes, predictable load)
  - Artifact Registry (Docker image storage)

Qdrant migration path: replace FAISS in archetype_store.py with
Qdrant client for horizontal scaling across multiple replicas.

Usage:
    uvicorn deploy.gcp.service:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tsfm_autoresearch.autoresearch import AutoresearchHarness
from tsfm_autoresearch.losses import SLATier
from tsfm_autoresearch.tsfm_client import TSFMClient

logger = logging.getLogger(__name__)

# ── Global state (loaded at startup) ─────────────────────────────────

_client: TSFMClient | None = None
_harness: AutoresearchHarness | None = None
_startup_time_s: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TimesFM at startup, clean up at shutdown."""
    global _client, _harness, _startup_time_s

    logger.info("Loading TimesFM 2.5 (frozen)...")
    t0 = time.perf_counter()
    _client = TSFMClient(
        max_context=512,
        max_horizon=128,
        per_core_batch_size=8,  # GPU-optimized batching
        torch_compile=True,     # JIT for lower latency
    )
    _harness = AutoresearchHarness(_client, default_K=8)
    _startup_time_s = time.perf_counter() - t0
    logger.info("Ready in %.1fs", _startup_time_s)

    yield

    logger.info("Shutting down")
    _client = None
    _harness = None


app = FastAPI(
    title="TSFM Autoresearch Service",
    description="Per-request autoresearch over frozen TimesFM for multi-tenant forecasting",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Models ──────────────────────────────────────────────────────────


class ForecastRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier")
    history: list[list[float]] = Field(
        ..., description="Multivariate history, shape (T, 4): [cpu, mem, net, disk]"
    )
    horizon: int = Field(60, ge=1, le=128, description="Forecast steps")
    sla_tier: str = Field("standard", pattern="^(premium|standard|basic)$")
    k: int = Field(8, ge=1, le=32, description="Configs per request")


class ForecastResponse(BaseModel):
    tenant_id: str
    point_forecast: list[list[float]]  # (horizon, 4)
    winning_config: dict
    sla_tier: str
    total_latency_ms: float
    model_version: str = "timesfm-2.5-200m"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    startup_time_s: float
    uptime_s: float


# ── Routes ──────────────────────────────────────────────────────────

_start_time = time.time()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if _client is not None else "starting",
        model_loaded=_client is not None,
        startup_time_s=_startup_time_s,
        uptime_s=time.time() - _start_time,
    )


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    if _harness is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    import numpy as np

    history = np.array(request.history, dtype=np.float64)
    if history.ndim != 2 or history.shape[1] != 4:
        raise HTTPException(
            status_code=400,
            detail=f"History must be (T, 4), got {history.shape}",
        )

    try:
        sla_tier = SLATier(request.sla_tier)
        response = _harness.forecast(
            tenant_id=request.tenant_id,
            history=history,
            horizon=request.horizon,
            sla_tier=sla_tier,
            K=request.k,
        )
    except Exception as e:
        logger.exception("Forecast failed for %s", request.tenant_id)
        raise HTTPException(status_code=500, detail=str(e))

    return ForecastResponse(
        tenant_id=request.tenant_id,
        point_forecast=response.final_forecast.point.tolist(),
        winning_config={
            "context_len": response.winning_config.context_len,
            "quantiles": response.winning_config.quantiles,
        },
        sla_tier=request.sla_tier,
        total_latency_ms=response.total_latency_ms,
    )
