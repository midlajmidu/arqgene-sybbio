"""
backend/services/fba_service.py
---------------------------------
Service layer for FBA and pFBA analysis.

Thread-safety design
---------------------
COBRApy model objects in the registry are SHARED across requests.  To
prevent one request's solver setting from corrupting a concurrent request
on the same model, every solve follows this pattern:

    with model:                              # --- outer isolation boundary ---
        configure_solver_in_context(...)     # mutation tracked, rolled back on exit
        result = _run_fba(model)             # inner `with model:` is nested — valid
    # context exits → solver restored → registry model is pristine again

`configure_solver_in_context` is called INSIDE the `with model:` block.
It was previously called outside (race condition fixed here).

Scientific purpose
-------------------
FBA:   max  c^T v   s.t.  S v = 0,  lb ≤ v ≤ ub
pFBA:  additionally minimises ∑|v_i| constrained to objective ≥ z*
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cobra

from backend.exceptions import ModelNotFoundError
from backend.schemas.requests import FBARequest
from backend.schemas.responses import (
    ErrorDetail,
    FBAResponse,
    FluxReaction,
    PFBAResponse,
)
from backend.services.model_registry import get_registry
from backend.utils.solve_utils import configure_solver_in_context
from backend.services.medium_service import apply_medium_from_metadata
from core.diagnostics import _run_fba, _run_pfba

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FBA
# ---------------------------------------------------------------------------


def run_fba(request: FBARequest) -> FBAResponse:
    """
    Execute standard FBA on a model from the registry.

    Raises
    ------
    ModelNotFoundError
        If `request.model_id` is not present in the registry.
        The route layer catches this and returns HTTP 404.
    """
    model = _fetch_model(request.model_id)

    # ── Isolation boundary ─────────────────────────────────────────────────
    # solver configuration AND the solve both happen inside this context.
    # COBRApy's History stack rolls back model.solver when the block exits,
    # so the shared registry instance is left untouched after each request.
    # ──────────────────────────────────────────────────────────────────────
    with model:
        configure_solver_in_context(
            model,
            request.solver,
            request.feasibility_tol,
            request.optimality_tol,
        )
        apply_medium_from_metadata(model, request.model_id)  # ← medium overlay (rolled back on exit)
        # _run_fba opens its own inner `with model:` — nested contexts are
        # fully supported by COBRApy (>= 0.25).
        result = _run_fba(model)

    top_rxns = _to_flux_reactions(result.top_reactions)

    return FBAResponse(
        success=result.is_optimal,
        message=(
            "FBA completed successfully."
            if result.is_optimal
            else f"FBA returned non-optimal status: {result.status}"
        ),
        model_id=request.model_id,
        analysis_type="FBA",
        solver_status=result.status,
        solver_name=result.solver_name,
        objective_value=result.objective_value,
        growth_rate=result.growth_rate,
        top_reactions=_to_flux_reactions(result.top_reactions),
        exchange_fluxes=_to_exchange_fluxes(result.exchange_fluxes),
        error=(
            ErrorDetail(code="SOLVER_NON_OPTIMAL", message=result.error_message)
            if result.error_message
            else None
        ),
    )


# ---------------------------------------------------------------------------
# pFBA
# ---------------------------------------------------------------------------


def run_pfba(request: FBARequest) -> PFBAResponse:
    """
    Execute Parsimonious FBA on a model from the registry.

    pFBA = two-step LP:
      Step 1: maximise objective  → z*
      Step 2: minimise ∑|v_i|     subject to objective ≥ z*
                                  (handled internally by COBRApy)

    Raises
    ------
    ModelNotFoundError
        If `request.model_id` is not present in the registry.
    """
    model = _fetch_model(request.model_id)

    with model:
        configure_solver_in_context(
            model,
            request.solver,
            request.feasibility_tol,
            request.optimality_tol,
        )
        apply_medium_from_metadata(model, request.model_id)  # ← medium overlay
        result = _run_pfba(model)

    total_abs_flux = 0.0
    if result.fluxes is not None:
        # L1-norm of the full flux vector — pFBA's secondary minimisation target
        total_abs_flux = float(result.fluxes.abs().sum())

    top_rxns = _to_flux_reactions(result.top_reactions)

    return PFBAResponse(
        success=result.is_optimal,
        message=(
            "pFBA completed successfully."
            if result.is_optimal
            else f"pFBA returned non-optimal status: {result.status}"
        ),
        model_id=request.model_id,
        analysis_type="pFBA",
        solver_status=result.status,
        solver_name=result.solver_name,
        objective_value=result.objective_value,
        growth_rate=result.growth_rate,
        total_absolute_flux=total_abs_flux,
        top_reactions=_to_flux_reactions(result.top_reactions),
        exchange_fluxes=_to_exchange_fluxes(result.exchange_fluxes),
        error=(
            ErrorDetail(code="SOLVER_NON_OPTIMAL", message=result.error_message)
            if result.error_message
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_model(model_id: str) -> cobra.Model:
    """
    Retrieve a model or raise ModelNotFoundError.

    We raise rather than returning None so callers don't need to
    repeat the None-check pattern — the exception propagates cleanly
    through run_in_executor back to the route handler.
    """
    model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)
    return model


def _to_flux_reactions(df) -> List[FluxReaction]:
    """Convert a pandas DataFrame of top fluxes to Pydantic FluxReaction list."""
    if df is None or df.empty:
        return []
    return [
        FluxReaction(
            reaction_id=str(row.get("Reaction ID", "")),
            equation=str(row.get("Equation", "")),
            flux=float(row.get("Flux (mmol/gDW/h)", 0.0)),
            abs_flux=float(row.get("|Flux|", 0.0)),
            subsystem=str(row.get("Subsystem", "N/A")),
        )
        for _, row in df.iterrows()
    ]


def _to_exchange_fluxes(df) -> List[ExchangeFlux]:
    """Convert a pandas DataFrame of exchange fluxes to Pydantic ExchangeFlux list."""
    if df is None or df.empty:
        return []
    from backend.schemas.responses import ExchangeFlux
    return [
        ExchangeFlux(
            metabolite_name=str(row.get("metabolite_name", "")),
            reaction_id=str(row.get("reaction_id", "")),
            flux=float(row.get("flux", 0.0)),
            direction=str(row.get("direction", "")),
        )
        for _, row in df.iterrows()
    ]
