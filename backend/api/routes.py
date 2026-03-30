"""
backend/api/routes.py
-----------------------
FastAPI router — all STEP 1 API endpoints.

Changes from initial version
------------------------------
1.  asyncio.get_running_loop()  — replaces deprecated get_event_loop()
2.  asyncio.wait_for(timeout=N) — every executor call has a wall-clock limit
3.  HTTP 408 on timeout         — structured JSON, server does NOT crash
4.  HTTP 404 on missing model   — ModelNotFoundError caught here
5.  HTTP 422 on bad solver      — SolverNotAllowedError + Pydantic validation
6.  HTTP 413 on oversized file  — enforced before bytes reach the parser
7.  Solver allowlist            — validated before executor call
8.  ThreadPoolExecutor shutdown — registered in lifespan (see main.py)

Timeout note
------------
asyncio.wait_for() makes the awaiting coroutine return early, but the
underlying ThreadPoolExecutor thread cannot be interrupted (Python
limitation).  The thread continues and discards its result.  Future
improvement: ProcessPoolExecutor + SIGALRM for true solver cancellation.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.exceptions import ModelNotFoundError, SolverNotAllowedError, ModelTooLargeError, InfeasibleModelError, ReactionNotFoundError, ComputationTooExpensiveError
from backend.schemas.requests import FBARequest, ValidationRequest, FVARequest, ObjectiveSwitchRequest, ProductionEnvelopeRequest, MediumUpdateRequest, PresetRequest, OptKnockRequest
from backend.schemas.responses import (
    FBAResponse,
    FVAResponse,
    GrowthAuditResponse,
    MediumResponse,
    ModelSummaryResponse,
    ObjectiveSwitchResponse,
    OptKnockResponse,
    PFBAResponse,
    ProductionEnvelopeResponse,
    ReactionsListResponse,
    ValidationResponse,
)
from backend.services.fba_service import run_fba, run_pfba
from backend.services.fva_service import run_fva
from backend.services.production_service import run_objective_switch, run_production_envelope
from backend.services.growth_audit_service import run_growth_diagnostic
from backend.services.medium_service import (
    apply_preset,
    get_medium,
    reset_medium,
    update_medium,
)
from backend.services.model_registry import get_registry
from backend.services.model_service import upload_and_register_model
from backend.services.reactions_service import list_reactions
from backend.services.validation_service import run_validation
from backend.services.optknock_service import run_optknock

_MEDIUM_SHORT_TIMEOUT = 30.0  # medium ops are metadata-only; no LP needed
from backend.utils.solve_utils import (
    DEFAULT_SOLVER_TIMEOUT,
    MAX_UPLOAD_BYTES,
    VALIDATION_SOLVER_TIMEOUT,
    validate_solver,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Thread pool for CPU-bound COBRApy computations.
# max_workers=4 handles 4 concurrent FBA solves; FVA validation saturates one
# worker for minutes on large models (see VALIDATION_SOLVER_TIMEOUT).
# Exposed at module level so main.py lifespan can shut it down cleanly.
# ---------------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cobra_worker")


# ---------------------------------------------------------------------------
# Error response helpers
# ---------------------------------------------------------------------------

def _timeout_json(operation: str, timeout: float) -> dict:
    return {
        "success": False,
        "message": f"{operation} timed out after {timeout:.0f} seconds.",
        "error": {
            "code": "SOLVER_TIMEOUT",
            "message": (
                f"The {operation} solver exceeded the {timeout:.0f}s wall-clock limit. "
                "Try a smaller model or disable FVA."
            ),
        },
    }


def _not_found_json(model_id: str) -> dict:
    return {
        "success": False,
        "message": f"Model '{model_id}' not found.",
        "error": {
            "code": "MODEL_NOT_FOUND",
            "message": (
                f"No model is registered with UUID '{model_id}'. "
                "Upload a model first via POST /api/v1/upload-model."
            ),
        },
    }


def _solver_invalid_json(solver: str) -> dict:
    from backend.utils.solve_utils import ALLOWED_SOLVERS
    return {
        "success": False,
        "message": f"Solver '{solver}' is not allowed.",
        "error": {
            "code": "SOLVER_NOT_ALLOWED",
            "message": f"Allowed solvers: {ALLOWED_SOLVERS}",
        },
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    """Return server status and registry size."""
    registry = get_registry()
    return {
        "status": "ok",
        "registered_models": len(registry),
        "executor_workers": executor._max_workers,
    }


# ---------------------------------------------------------------------------
# POST /upload-model
# ---------------------------------------------------------------------------

@router.post(
    "/upload-model",
    response_model=ModelSummaryResponse,
    summary="Upload and parse an SBML metabolic model",
    tags=["Model"],
    responses={
        413: {"description": "File too large (> 50 MB)"},
        422: {"description": "Invalid solver name"},
        408: {"description": "Parse timeout"},
    },
)
async def upload_model(
    file: UploadFile = File(..., description="SBML (.xml) model file"),
    solver: str = Form("glpk", description="LP solver backend"),
    feasibility_tol: float = Form(1e-7),
    optimality_tol: float = Form(1e-7),
) -> Union[ModelSummaryResponse, JSONResponse]:
    """
    Accept an SBML upload, parse, configure solver, register in memory.

    Returns a UUID (`model_id`) for all subsequent API calls.
    """
    # ---- 1. File size guard ----
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "message": f"File exceeds the {mb} MB size limit.",
                "error": {
                    "code": "FILE_TOO_LARGE",
                    "message": f"Upload is {len(file_bytes) // (1024*1024)} MB; limit is {mb} MB.",
                },
            },
        )

    # ---- 2. Solver allowlist ----
    try:
        validate_solver(solver)
    except SolverNotAllowedError:
        return JSONResponse(status_code=422, content=_solver_invalid_json(solver))

    logger.info(
        "Model upload: file='%s' size=%d bytes solver=%s",
        file.filename, len(file_bytes), solver,
    )

    loop = asyncio.get_running_loop()
    try:
        response: ModelSummaryResponse = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(
                    upload_and_register_model,
                    file_bytes,
                    file.filename or "model.xml",
                    solver,
                    feasibility_tol,
                    optimality_tol,
                ),
            ),
            timeout=DEFAULT_SOLVER_TIMEOUT,
        )
        return response
    except asyncio.TimeoutError:
        logger.error(
            "Model upload/parse timed out (%.0fs) for file='%s'",
            DEFAULT_SOLVER_TIMEOUT, file.filename,
        )
        return JSONResponse(
            status_code=408,
            content=_timeout_json("Model parsing", DEFAULT_SOLVER_TIMEOUT),
        )


# ---------------------------------------------------------------------------
# POST /run-fba
# ---------------------------------------------------------------------------

@router.post(
    "/run-fba",
    response_model=FBAResponse,
    summary="Run standard Flux Balance Analysis",
    tags=["Analysis"],
    responses={
        404: {"description": "Model not found"},
        408: {"description": "Solver timeout"},
        422: {"description": "Invalid solver"},
    },
)
async def api_run_fba(
    request: FBARequest,
) -> Union[FBAResponse, JSONResponse]:
    """
    FBA: maximise/minimise objective on a registered model.

    Growth rate, objective value, solver status, and top-10 flux reactions.

    Scientific note: solver config happens INSIDE `with model:` in the
    service layer — no shared state is mutated between requests.
    """
    try:
        validate_solver(request.solver)
    except SolverNotAllowedError:
        return JSONResponse(status_code=422, content=_solver_invalid_json(request.solver))

    logger.info("FBA: model_id=%s solver=%s", request.model_id, request.solver)

    loop = asyncio.get_running_loop()
    try:
        response: FBAResponse = await asyncio.wait_for(
            loop.run_in_executor(executor, partial(run_fba, request)),
            timeout=DEFAULT_SOLVER_TIMEOUT,
        )
        return response
    except asyncio.TimeoutError:
        logger.error(
            "FBA timed out (%.0fs) for model_id=%s", DEFAULT_SOLVER_TIMEOUT, request.model_id
        )
        return JSONResponse(
            status_code=408,
            content=_timeout_json("FBA", DEFAULT_SOLVER_TIMEOUT),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))


# ---------------------------------------------------------------------------
# POST /run-pfba
# ---------------------------------------------------------------------------

@router.post(
    "/run-pfba",
    response_model=PFBAResponse,
    summary="Run parsimonious Flux Balance Analysis",
    tags=["Analysis"],
    responses={
        404: {"description": "Model not found"},
        408: {"description": "Solver timeout"},
        422: {"description": "Invalid solver"},
    },
)
async def api_run_pfba(
    request: FBARequest,
) -> Union[PFBAResponse, JSONResponse]:
    """
    pFBA: FBA + L1-norm minimisation (two LP solves, ~2× slower than FBA).

    Returns all FBA fields plus `total_absolute_flux` (L1-norm of full
    flux vector — measures total metabolic activity).
    """
    try:
        validate_solver(request.solver)
    except SolverNotAllowedError:
        return JSONResponse(status_code=422, content=_solver_invalid_json(request.solver))

    logger.info("pFBA: model_id=%s solver=%s", request.model_id, request.solver)

    loop = asyncio.get_running_loop()
    try:
        response: PFBAResponse = await asyncio.wait_for(
            loop.run_in_executor(executor, partial(run_pfba, request)),
            timeout=DEFAULT_SOLVER_TIMEOUT,
        )
        return response
    except asyncio.TimeoutError:
        logger.error(
            "pFBA timed out (%.0fs) for model_id=%s", DEFAULT_SOLVER_TIMEOUT, request.model_id
        )
        return JSONResponse(
            status_code=408,
            content=_timeout_json("pFBA", DEFAULT_SOLVER_TIMEOUT),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))


# ---------------------------------------------------------------------------
# POST /validate-model
# ---------------------------------------------------------------------------

@router.post(
    "/validate-model",
    response_model=ValidationResponse,
    summary="Run full model validation suite",
    tags=["Validation"],
    responses={
        404: {"description": "Model not found"},
        408: {"description": "Validation timeout (FVA on large model)"},
    },
)
async def api_validate_model(
    request: ValidationRequest,
) -> Union[ValidationResponse, JSONResponse]:
    """
    Full validation suite (5 checks):

    1. Objective feasibility (LP solve)
    2. Inconsistent bounds (lb > ub, structural)
    3. Gene-orphan reactions (structural)
    4. Blocked reactions via FVA (optional — disable with `run_fva=false`)
    5. Stoichiometric mass balance (formula-annotated reactions)

    FVA timeout: 600 s (longer than FBA — FVA solves 2N LPs).
    Set `run_fva=false` for genome-scale models (> 3 000 reactions).
    """
    logger.info(
        "Validation: model_id=%s run_fva=%s", request.model_id, request.run_fva
    )

    loop = asyncio.get_running_loop()
    timeout = VALIDATION_SOLVER_TIMEOUT if request.run_fva else DEFAULT_SOLVER_TIMEOUT
    try:
        response: ValidationResponse = await asyncio.wait_for(
            loop.run_in_executor(executor, partial(run_validation, request)),
            timeout=timeout,
        )
        return response
    except asyncio.TimeoutError:
        logger.error(
            "Validation timed out (%.0fs) for model_id=%s", timeout, request.model_id
        )
        return JSONResponse(
            status_code=408,
            content=_timeout_json("Validation", timeout),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))


# ---------------------------------------------------------------------------
# GET /reactions/{model_id}
# ---------------------------------------------------------------------------

@router.get(
    "/reactions/{model_id}",
    response_model=ReactionsListResponse,
    summary="Paginated reaction list",
    tags=["Model"],
    responses={
        404: {"description": "Model not found"},
    },
)
async def api_list_reactions(
    model_id: str,
    page: int = 1,
    page_size: int = 25,
    search: Optional[str] = None,
    subsystem: Optional[str] = None,
) -> Union[ReactionsListResponse, JSONResponse]:
    """
    Paginated, searchable reaction list.  No LP solve — fast read-only.

    Query params: page, page_size (1–200), search (free text), subsystem (exact).
    """
    loop = asyncio.get_running_loop()
    try:
        response: ReactionsListResponse = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(list_reactions, model_id, page, page_size, search, subsystem),
            ),
            timeout=30.0,  # read-only, should be fast
        )
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content=_timeout_json("Reaction listing", 30.0),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))


# ---------------------------------------------------------------------------
# POST /models/{model_id}/fva
# ---------------------------------------------------------------------------

@router.post(
    "/models/{model_id}/fva",
    response_model=FVAResponse,
    summary="Run Flux Variability Analysis",
    tags=["Analysis"],
    responses={
        400: {"description": "Large model — set confirm_full_model=True to proceed"},
        404: {"description": "Model not found"},
        408: {"description": "FVA solver timeout"},
        422: {"description": "Invalid solver or infeasible model"},
    },
)
async def api_run_fva(
    model_id: str,
    request: FVARequest,
) -> Union[FVAResponse, JSONResponse]:
    """
    Flux Variability Analysis: compute the minimum and maximum steady-state
    flux for each reaction compatible with `fraction_of_optimum` × z*.

    Requires 2 × |R| LP solves.  For |R| > 2 000, set `confirm_full_model=True`
    or pass an explicit `reaction_ids` subset.

    Timeout: 600 s (configurable per-request up to that ceiling).
    """
    try:
        validate_solver(request.solver)
    except SolverNotAllowedError:
        return JSONResponse(status_code=422, content=_solver_invalid_json(request.solver))

    # Per-request timeout, capped at the server maximum
    timeout = (
        min(float(request.timeout), VALIDATION_SOLVER_TIMEOUT)
        if request.timeout
        else VALIDATION_SOLVER_TIMEOUT
    )

    logger.info(
        "FVA: model_id=%s solver=%s fraction=%.2f reactions=%s confirm=%s timeout=%.0fs",
        model_id, request.solver, request.fraction_of_optimum,
        len(request.reaction_ids) if request.reaction_ids else "all",
        request.confirm_full_model, timeout,
    )

    loop = asyncio.get_running_loop()
    try:
        response: FVAResponse = await asyncio.wait_for(
            loop.run_in_executor(executor, partial(run_fva, request, model_id)),
            timeout=timeout,
        )
        return response
    except asyncio.TimeoutError:
        logger.error(
            "FVA timed out (%.0fs) for model_id=%s", timeout, model_id
        )
        return JSONResponse(
            status_code=408,
            content=_timeout_json("FVA", timeout),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))
    except ModelTooLargeError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "MODEL_TOO_LARGE",
                "message": (
                    f"Model has {exc.n_reactions} reactions. "
                    f"Full FVA may take several minutes. "
                    f"Set confirm_full_model=True to proceed."
                ),
            },
        )
    except InfeasibleModelError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "INFEASIBLE_MODEL",
                "message": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────
# POST /models/{model_id}/objective
# ─────────────────────────────────────────────────────────────────


def _reaction_not_found_json(reaction_id: str) -> dict:
    return {
        "success": False,
        "message": f"Reaction '{reaction_id}' not found.",
        "error": {
            "code": "REACTION_NOT_FOUND",
            "message": f"Reaction '{reaction_id}' does not exist in this model.",
        },
    }


def _too_expensive_json(message: str) -> dict:
    return {
        "success": False,
        "message": message,
        "error": {"code": "COMPUTATION_TOO_EXPENSIVE", "message": message},
    }


def _infeasible_json(message: str) -> dict:
    return {
        "success": False,
        "message": message,
        "error": {"code": "INFEASIBLE_MODEL", "message": message},
    }


@router.post(
    "/models/{model_id}/objective",
    response_model=ObjectiveSwitchResponse,
    summary="Switch optimization objective",
    tags=["Analysis"],
    responses={
        404: {"description": "Model or reaction not found"},
        408: {"description": "Solver timeout"},
        422: {"description": "Objective infeasible"},
    },
)
async def api_set_objective(
    model_id: str,
    request: ObjectiveSwitchRequest,
) -> Union[ObjectiveSwitchResponse, JSONResponse]:
    """
    Switch the LP objective to any reaction in the model and return
    the baseline flux for validation.

    The shared registry model is NOT permanently mutated.
    The choice is stored in registry metadata for use by the envelope endpoint.
    """
    logger.info(
        "Objective switch: model_id=%s reaction=%s direction=%s",
        model_id, request.reaction_id, request.direction,
    )
    loop = asyncio.get_running_loop()
    try:
        response: ObjectiveSwitchResponse = await asyncio.wait_for(
            loop.run_in_executor(
                executor, partial(run_objective_switch, request, model_id)
            ),
            timeout=DEFAULT_SOLVER_TIMEOUT,
        )
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content=_timeout_json("Objective validation", DEFAULT_SOLVER_TIMEOUT),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))
    except ReactionNotFoundError as exc:
        return JSONResponse(status_code=404, content=_reaction_not_found_json(exc.reaction_id))
    except InfeasibleModelError as exc:
        return JSONResponse(status_code=422, content=_infeasible_json(str(exc)))


# ─────────────────────────────────────────────────────────────────
# POST /models/{model_id}/production-envelope
# ─────────────────────────────────────────────────────────────────


@router.post(
    "/models/{model_id}/production-envelope",
    response_model=ProductionEnvelopeResponse,
    summary="Compute growth–production Pareto envelope",
    tags=["Analysis"],
    responses={
        400: {"description": "Computation too expensive"},
        404: {"description": "Model or reaction not found"},
        408: {"description": "Envelope solver timeout"},
        422: {"description": "Biomass LP infeasible"},
    },
)
async def api_production_envelope(
    model_id: str,
    request: ProductionEnvelopeRequest,
) -> Union[ProductionEnvelopeResponse, JSONResponse]:
    """
    Growth–production envelope: scan biomass flux from 0 to max growth,
    optimising product flux at each point.

    Requires `steps` + 1 LP solves.  For large models reduce `steps`.
    Timeout: 600 s.
    """
    try:
        validate_solver(request.solver)
    except SolverNotAllowedError:
        return JSONResponse(status_code=422, content=_solver_invalid_json(request.solver))

    logger.info(
        "Envelope: model_id=%s biomass=%s product=%s steps=%d",
        model_id, request.biomass_reaction, request.product_reaction, request.steps,
    )
    loop = asyncio.get_running_loop()
    try:
        response: ProductionEnvelopeResponse = await asyncio.wait_for(
            loop.run_in_executor(
                executor, partial(run_production_envelope, request, model_id)
            ),
            timeout=VALIDATION_SOLVER_TIMEOUT,
        )
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content=_timeout_json("Production envelope", VALIDATION_SOLVER_TIMEOUT),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))
    except ReactionNotFoundError as exc:
        return JSONResponse(status_code=404, content=_reaction_not_found_json(exc.reaction_id))
    except InfeasibleModelError as exc:
        return JSONResponse(status_code=422, content=_infeasible_json(str(exc)))
    except ComputationTooExpensiveError as exc:
        return JSONResponse(status_code=400, content=_too_expensive_json(str(exc)))


# ─────────────────────────────────────────────────────────────────
# Medium configuration — 4 endpoints
# ─────────────────────────────────────────────────────────────────


@router.get(
    "/models/{model_id}/medium",
    response_model=MediumResponse,
    summary="Get current medium configuration",
    tags=["Medium"],
    responses={404: {"description": "Model not found"}},
)
async def api_get_medium(model_id: str) -> Union[MediumResponse, JSONResponse]:
    """Return all exchange reactions with original and effective bounds."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, partial(get_medium, model_id)),
            timeout=_MEDIUM_SHORT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=408, content=_timeout_json("Get medium", _MEDIUM_SHORT_TIMEOUT))
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))


@router.post(
    "/models/{model_id}/medium",
    response_model=MediumResponse,
    summary="Update exchange bounds in medium",
    tags=["Medium"],
    responses={
        404: {"description": "Model or reaction not found"},
        422: {"description": "Non-exchange reaction or too many updates"},
    },
)
async def api_update_medium(
    model_id: str,
    request: MediumUpdateRequest,
) -> Union[MediumResponse, JSONResponse]:
    """Merge bound updates into medium metadata. Does NOT mutate the cobra.Model."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, partial(update_medium, request, model_id)),
            timeout=_MEDIUM_SHORT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=408, content=_timeout_json("Update medium", _MEDIUM_SHORT_TIMEOUT))
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))
    except ReactionNotFoundError as exc:
        return JSONResponse(status_code=404, content=_reaction_not_found_json(exc.reaction_id))
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": str(exc), "error": {"code": "MEDIUM_VALIDATION_ERROR", "message": str(exc)}},
        )


@router.post(
    "/models/{model_id}/medium/preset",
    response_model=MediumResponse,
    summary="Apply a named medium preset",
    tags=["Medium"],
    responses={
        404: {"description": "Model not found"},
        422: {"description": "Unknown preset"},
    },
)
async def api_apply_preset(
    model_id: str,
    request: PresetRequest,
) -> Union[MediumResponse, JSONResponse]:
    """Overwrite medium with a preset (aerobic_glucose / anaerobic_glucose / minimal_closed)."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, partial(apply_preset, request.preset, model_id)),
            timeout=_MEDIUM_SHORT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=408, content=_timeout_json("Apply preset", _MEDIUM_SHORT_TIMEOUT))
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": str(exc), "error": {"code": "UNKNOWN_PRESET", "message": str(exc)}},
        )


@router.post(
    "/models/{model_id}/medium/reset",
    response_model=MediumResponse,
    summary="Reset medium to original SBML bounds",
    tags=["Medium"],
    responses={404: {"description": "Model not found"}},
)
async def api_reset_medium(model_id: str) -> Union[MediumResponse, JSONResponse]:
    """Clear all medium modifications; restores original SBML bounds for future solves."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, partial(reset_medium, model_id)),
            timeout=_MEDIUM_SHORT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=408, content=_timeout_json("Reset medium", _MEDIUM_SHORT_TIMEOUT))
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))


# ─────────────────────────────────────────────────────────────────
# GET /models/{model_id}/growth-audit
# ─────────────────────────────────────────────────────────────────


@router.get(
    "/models/{model_id}/growth-audit",
    response_model=GrowthAuditResponse,
    summary="Diagnose zero-growth causes",
    tags=["Analysis"],
    responses={
        404: {"description": "Model not found"},
        408: {"description": "Audit timeout"},
    },
)
async def api_growth_audit(
    model_id: str,
) -> Union[GrowthAuditResponse, JSONResponse]:
    """
    Scientific debugging endpoint: identify why FBA growth = 0.

    Runs 10 ordered checks:
      1. Biomass reaction structure
      2. Baseline FBA solve
      3–6. Carbon, nitrogen, phosphate/sulfate, oxygen exchange bounds
      7. ATPM maintenance constraint
      8. Exchange uptake census
      9. Single-reaction FVA (structural blockage)
     10. Root-cause inference

    Returns a `likely_cause` string with the most probable explanation.
    All LP work happens inside `with model:` — registry model not mutated.
    Timeout: 300 s.
    """
    logger.info("Growth audit: model_id=%s", model_id)
    loop = asyncio.get_running_loop()
    try:
        response: GrowthAuditResponse = await asyncio.wait_for(
            loop.run_in_executor(executor, partial(run_growth_diagnostic, model_id)),
            timeout=DEFAULT_SOLVER_TIMEOUT,
        )
        return response
    except asyncio.TimeoutError:
        logger.error(
            "Growth audit timed out (%.0fs) for model_id=%s",
            DEFAULT_SOLVER_TIMEOUT, model_id,
        )
        return JSONResponse(
            status_code=408,
            content=_timeout_json("Growth audit", DEFAULT_SOLVER_TIMEOUT),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))


# ─────────────────────────────────────────────────────────────────
# GET /models
# ─────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------

@router.get("/models", summary="List registered models", tags=["Model"])
async def list_models() -> dict:
    """Return all model UUIDs currently held in the in-memory registry."""
    registry = get_registry()
    return {"model_ids": registry.list_ids(), "count": len(registry)}


# ---------------------------------------------------------------------------
# DELETE /models/{model_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/models/{model_id}",
    summary="Remove a model from registry",
    tags=["Model"],
    responses={404: {"description": "Model not found"}},
)
async def delete_model(model_id: str) -> dict:
    """Explicitly free a model's memory from the registry."""
    removed = get_registry().remove(model_id)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found in registry.",
        )
    return {"success": True, "message": f"Model '{model_id}' removed from registry."}


# ─────────────────────────────────────────────────────────────────
# POST /models/{model_id}/optknock
# ─────────────────────────────────────────────────────────────────


@router.post(
    "/models/{model_id}/optknock",
    response_model=OptKnockResponse,
    summary="OptKnock-style greedy strain design",
    tags=["Analysis"],
    responses={
        400: {"description": "Computation too expensive"},
        404: {"description": "Model or reaction not found"},
        408: {"description": "Solver timeout"},
        422: {"description": "Infeasible model or invalid configuration"},
    },
)
async def api_optknock(
    model_id: str,
    request: OptKnockRequest,
) -> Union[OptKnockResponse, JSONResponse]:
    """
    Greedy OptKnock strain design: find reaction knockouts that maximise
    product flux while keeping growth ≥ growth_fraction × baseline.

    Algorithm: greedy sequential search (1‬200 FBA solves for k≤1, ≤30 min).
    Timeout: 600 s (configurable via request.timeout).
    All LP work inside `with model:` — registry model never mutated.
    """
    logger.info(
        "OptKnock: model_id=%s biomass=%s product=%s max_ko=%d gf=%.2f",
        model_id, request.biomass_reaction, request.product_reaction,
        request.max_knockouts, request.growth_fraction,
    )
    timeout = float(min(request.timeout or 600, 600))
    loop = asyncio.get_running_loop()
    try:
        response: OptKnockResponse = await asyncio.wait_for(
            loop.run_in_executor(executor, partial(run_optknock, request, model_id)),
            timeout=timeout,
        )
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content=_timeout_json("OptKnock", timeout),
        )
    except ModelNotFoundError as exc:
        return JSONResponse(status_code=404, content=_not_found_json(exc.model_id))
    except ReactionNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": f"Reaction '{exc.reaction_id}' not found in model.",
                "error": {"code": "REACTION_NOT_FOUND", "message": str(exc)},
            },
        )
    except InfeasibleModelError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": str(exc),
                "error": {"code": "INFEASIBLE_MODEL", "message": str(exc)},
            },
        )
    except ComputationTooExpensiveError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": str(exc),
                "error": {"code": "COMPUTATION_TOO_EXPENSIVE", "message": str(exc)},
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": str(exc)},
        )


# ─────────────────────────────────────────────────────────────────
# Genome-to-Model Reconstruction
# ─────────────────────────────────────────────────────────────────


@router.post(
    "/upload-genome",
    summary="Upload a genome (.fna/.faa) and start reconstruction",
    tags=["Genome"],
    responses={
        413: {"description": "File too large"},
        422: {"description": "Invalid file format"},
    },
)
async def api_upload_genome(
    file: UploadFile = File(..., description="FASTA file (.fna or .faa)"),
    solver: str = Form("glpk", description="LP solver for the final model"),
) -> JSONResponse:
    """
    Accept a FASTA genome/proteome upload and start async reconstruction.

    Returns a job_id that can be polled via GET /genome-job/{job_id}.
    """
    from backend.services.genome_service import start_genome_reconstruction

    file_bytes = await file.read()

    # Size guard
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "message": f"File exceeds {mb} MB limit.",
            },
        )

    # Basic FASTA validation
    filename = file.filename or "sequences.faa"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    valid_exts = {"fna", "faa", "fa", "fasta", "pep", "prot", "ffn", "frn"}
    if ext not in valid_exts:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": f"Unsupported file extension '.{ext}'. Expected: {', '.join(sorted(valid_exts))}",
            },
        )

    # Check content starts with >
    text_preview = file_bytes[:200].decode("utf-8", errors="replace").strip()
    if not text_preview.startswith(">"):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "File does not appear to be valid FASTA format (must start with '>').",
            },
        )

    logger.info("Genome upload: file='%s' size=%d bytes", filename, len(file_bytes))

    # Start async reconstruction
    job_id = start_genome_reconstruction(
        file_bytes=file_bytes,
        filename=filename,
        solver=solver,
    )

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "message": "Genome reconstruction started.",
            "job_id": job_id,
        },
    )


@router.get(
    "/genome-job/{job_id}",
    summary="Poll genome reconstruction job status",
    tags=["Genome"],
    responses={
        404: {"description": "Job not found"},
    },
)
async def api_genome_job_status(job_id: str) -> JSONResponse:
    """
    Poll the status of a genome reconstruction job.

    Returns status, progress (0.0–1.0), message, and on completion
    the model_id for use with all existing simulation endpoints.
    """
    from backend.services.genome_service import get_job_store

    job = get_job_store().get_job(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"Job '{job_id}' not found."},
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "job_id": job.job_id,
            "filename": job.filename,
            "status": job.status,
            "progress": round(job.progress, 3),
            "message": job.message,
            "model_id": job.model_id,
            "report": job.report,
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        },
    )
