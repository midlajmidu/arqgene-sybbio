"""
backend/services/fva_service.py
---------------------------------
Service layer for Flux Variability Analysis (FVA).

Scientific purpose
-------------------
FVA determines the minimum and maximum flux through each reaction
that is compatible with a given fraction of the optimal objective:

    For each reaction j:
        min / max  v_j
        s.t.  S v = 0
              lb ≤ v ≤ ub
              c^T v ≥ fraction_of_optimum × z*

where z* is the FBA-optimal objective value.

This requires solving 2 × |R| LP problems (one min, one max per reaction),
making it O(N) times more expensive than a single FBA solve.

Thread-safety pattern
----------------------
All LP work (baseline FBA + FVA) runs inside a single outer `with model:`
block after calling configure_solver_in_context().  This ensures the
shared registry model's solver and bounds are rolled back on context exit,
consistent with the FBA/pFBA service pattern.

Confirmation gate
------------------
Full-model FVA on > 2 000 reactions can take 5–30 minutes with GLPK.
A ModelTooLargeError is raised when the client hasn't acknowledged this
via `confirm_full_model=True`.

Custom reaction subset
-----------------------
If `reaction_ids` is provided, only those reactions are analysed.
Unknown IDs are silently skipped (COBRApy ignores them too).  If none
of the provided IDs exist in the model, an early HTTP 422 is returned.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cobra
import cobra.flux_analysis

from backend.exceptions import (
    InfeasibleModelError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from backend.schemas.requests import FVARequest
from backend.schemas.responses import (
    ErrorDetail,
    FVAResponse,
    FVAReactionResult,
)
from backend.services.model_registry import get_registry
from backend.utils.solve_utils import configure_solver_in_context
from backend.services.medium_service import apply_medium_from_metadata

logger = logging.getLogger(__name__)

# Reactions whose absolute flux is below this threshold are considered blocked.
_BLOCKED_TOL: float = 1e-9
# Hard cap on explicitly requested reaction IDs to prevent oversized payloads.
_MAX_REACTION_IDS: int = 3_000


def run_fva(request: FVARequest, model_id: str) -> FVAResponse:
    """
    Execute Flux Variability Analysis on a registered model.

    Parameters
    ----------
    request : FVARequest
        Validated request payload.
    model_id : str
        UUID of the model in the registry (passed separately so it can be
        part of the URL path in the route rather than the JSON body).

    Returns
    -------
    FVAResponse
        Full FVA results including per-reaction min/max and blocked flag.

    Raises
    ------
    ModelNotFoundError
        Route catches → HTTP 404.
    ModelTooLargeError
        Route catches → HTTP 400 (user must confirm).
    InfeasibleModelError
        Route catches → HTTP 422.
    """
    model: cobra.Model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    n_total = len(model.reactions)
    reaction_list: Optional[List[cobra.Reaction]] = None

    # ------------------------------------------------------------------
    # 1. Determine reaction subset
    # ------------------------------------------------------------------
    if request.reaction_ids is not None:
        # Explicit subset requested
        if len(request.reaction_ids) > _MAX_REACTION_IDS:
            raise ModelTooLargeError(len(request.reaction_ids), limit=_MAX_REACTION_IDS)

        rxn_map = {r.id: r for r in model.reactions}
        reaction_list = [rxn_map[rid] for rid in request.reaction_ids if rid in rxn_map]

        if not reaction_list:
            return FVAResponse(
                success=False,
                message="None of the provided reaction_ids exist in this model.",
                model_id=model_id,
                total_reactions=n_total,
                error=ErrorDetail(
                    code="INVALID_REACTION_IDS",
                    message="No valid reaction IDs matched the model.",
                ),
            )
        logger.info(
            "FVA subset: %d / %d reactions requested for model_id=%s",
            len(reaction_list), n_total, model_id,
        )
    else:
        # Full-model FVA — require explicit confirmation on large models
        if n_total > 2_000 and not request.confirm_full_model:
            raise ModelTooLargeError(n_total)
        logger.info(
            "FVA full model: %d reactions, fraction=%.2f, model_id=%s",
            n_total, request.fraction_of_optimum, model_id,
        )

    # ------------------------------------------------------------------
    # 2. Run baseline FBA + FVA inside isolation context
    # ------------------------------------------------------------------
    # All LP work happens here. The outer `with model:` ensures solver
    # configuration is rolled back when this block exits, leaving the
    # shared registry model unchanged.
    with model:
        configure_solver_in_context(
            model,
            request.solver,
            # FVA uses default tolerances; no tolerance fields on FVARequest
            # to keep the schema minimal (add if needed in future).
        )
        apply_medium_from_metadata(model, model_id)  # ← medium overlay

        # ---- Baseline feasibility guard ----
        # FVA requires a feasible model. If the baseline is infeasible we
        # raise early — running FVA on an infeasible model would yield
        # all-zero or nonsensical ranges.
        baseline = model.optimize()
        if baseline.status != "optimal":
            raise InfeasibleModelError(
                f"Baseline FBA status: {baseline.status}. "
                f"FVA requires an optimal baseline."
            )

        objective_value = float(baseline.objective_value)

        # ---- FVA solve ----
        # processes=1: avoids forking a multiprocessing pool inside a
        # ThreadPoolExecutor thread (causes issues on macOS with GLPK).
        fva_df = cobra.flux_analysis.flux_variability_analysis(
            model,
            reaction_list=reaction_list,     # None = all reactions
            fraction_of_optimum=request.fraction_of_optimum,
            processes=1,
        )

        # Extract solver name from model (after configuration)
        try:
            solver_name = model.solver.interface.__name__.split(".")[-2]
        except Exception:
            solver_name = request.solver

    # ------------------------------------------------------------------
    # 3. Convert FVA DataFrame → response objects
    # ------------------------------------------------------------------
    # fva_df index = reaction IDs, columns = ["minimum", "maximum"]
    results: List[FVAReactionResult] = []
    for rxn_id, row in fva_df.iterrows():
        min_flux = float(row["minimum"])
        max_flux = float(row["maximum"])
        results.append(
            FVAReactionResult(
                reaction_id=str(rxn_id),
                minimum=min_flux,
                maximum=max_flux,
                range=max_flux - min_flux,
                is_blocked=(
                    abs(min_flux) < _BLOCKED_TOL
                    and abs(max_flux) < _BLOCKED_TOL
                ),
            )
        )

    blocked_count = sum(1 for r in results if r.is_blocked)

    logger.info(
        "FVA completed: model_id=%s analyzed=%d blocked=%d",
        model_id, len(results), blocked_count,
    )

    return FVAResponse(
        success=True,
        message=(
            f"FVA completed: {len(results):,} reactions analysed, "
            f"{blocked_count:,} blocked (fraction_of_optimum={request.fraction_of_optimum:.2f})."
        ),
        model_id=model_id,
        objective_value=objective_value,
        solver_status=baseline.status,
        solver_name=solver_name,
        fraction_of_optimum=request.fraction_of_optimum,
        total_reactions=n_total,
        analyzed_reactions=len(results),
        blocked_count=blocked_count,
        results=results,
    )
