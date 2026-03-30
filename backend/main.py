"""
backend/main.py
----------------
FastAPI application factory and server entrypoint for the SynB
Metabolic Engineering Platform — Step 1 backend.

Architecture:
    ┌────────────────────────────────────┐
    │  Streamlit Frontend (app.py)       │
    │  → HTTP requests via `requests`    │
    └────────────┬───────────────────────┘
                 │ JSON / multipart/form-data
    ┌────────────▼───────────────────────┐
    │  FastAPI  (this file)              │
    │  ├── /upload-model                 │
    │  ├── /run-fba                      │
    │  ├── /run-pfba                     │
    │  ├── /validate-model               │
    │  └── /reactions/{id}               │
    └────────────┬───────────────────────┘
                 │
    ┌────────────▼───────────────────────┐
    │  Service Layer                     │
    │  ├── model_service.py              │
    │  ├── fba_service.py                │
    │  ├── validation_service.py         │
    │  └── reactions_service.py          │
    └────────────┬───────────────────────┘
                 │
    ┌────────────▼───────────────────────┐
    │  Core Scientific Engine            │
    │  ├── core/model_loader.py          │
    │  ├── core/diagnostics.py           │
    │  ├── core/validation.py            │
    │  └── utils/solver_utils.py         │
    └────────────────────────────────────┘

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    OR
    python backend/main.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Routes module exposes the shared executor so the lifespan can shut it down.
from backend.api.routes import executor, router
from backend.services.model_registry import get_registry

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("synb.backend")


# ------------------------------------------------------------------
# Lifespan: startup / shutdown hooks
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialise registry, start eviction task, shut down executor."""
    logger.info("═══ SynB Backend starting up ═══")

    registry = get_registry()
    logger.info("Model registry ready (TTL=3600s)")
    logger.info("ThreadPoolExecutor ready (max_workers=%d)", executor._max_workers)

    # Background TTL eviction — runs every 10 minutes from the event loop.
    # evict_expired() is offloaded to a thread to avoid holding the event loop
    # while acquiring the registry RLock.
    async def _eviction_loop():
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(600)
            evicted = await loop.run_in_executor(None, registry.evict_expired)
            if evicted:
                logger.info("Eviction run: removed %d stale model(s).", evicted)

    task = asyncio.create_task(_eviction_loop())

    yield  # ← application runs here

    # ---- Graceful shutdown ----
    task.cancel()
    # Wait for in-flight solver threads to finish (up to 10 s) before process exits.
    executor.shutdown(wait=True, cancel_futures=False)
    logger.info("Executor shut down. ═══ SynB Backend stopped. ═══")


# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

app = FastAPI(
    title="SynB Metabolic Engineering API",
    description=(
        "Production backend for the SynB metabolic engineering SaaS platform. "
        "Step 1: SBML upload, model validation, and baseline FBA diagnostics.\n\n"
        "**Scientific foundation:** COBRApy + optlang LP/QP solvers.\n"
        "**Thread safety:** All COBRApy computations run in a ThreadPoolExecutor."
    ),
    version="1.0.0",
    contact={"name": "SynB Platform", "email": "support@synb.io"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# CORS — allow Streamlit frontend (localhost ports) to call this API
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",   # future React frontend
        "*",                       # dev convenience — restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Request timing middleware (useful for profiling large model runs)
# ------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Log and expose request processing time in milliseconds."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    logger.debug("%s %s → %d (%.1f ms)", request.method, request.url.path,
                 response.status_code, duration_ms)
    return response


# ------------------------------------------------------------------
# Global exception handler — no unhandled 500s leak to the client
# ------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected server error occurred.",
            "detail": str(exc),
        },
    )


# ------------------------------------------------------------------
# Mount API router (all endpoints prefixed with /api/v1)
# ------------------------------------------------------------------
app.include_router(router, prefix="/api/v1")


# ------------------------------------------------------------------
# Dev server entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
