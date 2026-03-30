"""
backend/services/production_service.py
----------------------------------------
Service layer for Objective Switching and Growth–Production Envelope analysis.

Scientific purpose
-------------------
Objective Switching
    Allows the user to redirect the LP objective from biomass to any exchange
    or internal reaction (e.g. ethanol export, succinate export) and verify
    that the model remains feasible under that objective.  The switch is
    validated INSIDE a `with model:` context so the shared registry instance
    is never permanently mutated.

Growth–Production Envelope (Phenotypic Phase Plane)
    Computes the Pareto frontier between cell growth and compound production.
    Algorithm:

        1. Maximise biomass → z_growth*
        2. Scan growth_range = linspace(0, z_growth*, steps)
        3. For each growth constraint g:
               biomass_rxn.lower_bound = g
               Maximise product_rxn  → record product flux
        4. Return all (growth, product) pairs

    All constraint mutations occur inside a single outer `with model:` block
    and are fully rolled back when it exits.  COBRApy's context manager
    tracks every bound and objective change on its History stack.

Thread-safety
    Each call gets an independent `with model:` context so concurrent requests
    on the same model_id are safely isolated at the bound level.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cobra
import numpy as np

from backend.exceptions import (
    ComputationTooExpensiveError,
    InfeasibleModelError,
    ModelNotFoundError,
    ReactionNotFoundError,
)
from backend.schemas.requests import ObjectiveSwitchRequest, ProductionEnvelopeRequest
from backend.schemas.responses import (
    ErrorDetail,
    ObjectiveSwitchResponse,
    ProductionEnvelopeResponse,
)
from backend.services.model_registry import get_registry
from backend.utils.solve_utils import configure_solver_in_context
from backend.services.medium_service import apply_medium_from_metadata

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Objective Switching
# ─────────────────────────────────────────────────────────────────


def run_objective_switch(
    request: ObjectiveSwitchRequest, model_id: str
) -> ObjectiveSwitchResponse:
    """
    Switch the optimization objective to `request.reaction_id` and validate
    feasibility by solving the LP once.

    The registry model is NEVER permanently mutated — the objective change
    lives inside `with model:` and is rolled back on context exit.

    The user's objective choice is persisted in registry metadata so the
    production envelope step can pre-fill the product reaction field.

    Raises
    ------
    ModelNotFoundError   → HTTP 404
    ReactionNotFoundError → HTTP 404
    InfeasibleModelError  → HTTP 422
    """
    model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    rxn_ids = {r.id for r in model.reactions}
    if request.reaction_id not in rxn_ids:
        raise ReactionNotFoundError(request.reaction_id)

    with model:
        configure_solver_in_context(model, "glpk")
        apply_medium_from_metadata(model, model_id)  # ← medium overlay

        # Set the new objective
        reaction = model.reactions.get_by_id(request.reaction_id)
        model.objective = reaction
        model.objective.direction = request.direction

        solution = model.optimize()
        if solution.status != "optimal":
            raise InfeasibleModelError(
                f"Objective '{request.reaction_id}' ({request.direction}) "
                f"is infeasible: solver status = {solution.status}."
            )

        obj_value = float(solution.objective_value)
        solver_name = _solver_name(model, "glpk")

    # Persist the objective choice in registry metadata (non-cobra data, safe)
    get_registry().set_model_metadata(
        model_id,
        objective_reaction=request.reaction_id,
        objective_direction=request.direction,
        objective_value=obj_value,
    )

    logger.info(
        "Objective switched: model_id=%s reaction=%s direction=%s value=%.4f",
        model_id, request.reaction_id, request.direction, obj_value,
    )

    return ObjectiveSwitchResponse(
        success=True,
        message=(
            f"Objective set to '{request.reaction_id}' "
            f"({request.direction}). Baseline value: {obj_value:.4f}."
        ),
        model_id=model_id,
        objective_reaction=request.reaction_id,
        direction=request.direction,
        baseline_objective_value=obj_value,
        solver_status=solution.status,
        solver_name=solver_name,
    )


# ─────────────────────────────────────────────────────────────────
# Growth–Production Envelope
# ─────────────────────────────────────────────────────────────────

# Models larger than this combined with many scan steps become slow
_LARGE_MODEL_REACTION_LIMIT = 5_000
_LARGE_MODEL_STEPS_LIMIT = 30


def run_production_envelope(
    request: ProductionEnvelopeRequest, model_id: str
) -> ProductionEnvelopeResponse:
    """
    Compute the growth–production Pareto envelope.

    Steps performed inside a single `with model:` isolation context:
      1. Configure solver
      2. Maximise biomass → z_growth*
      3. Scan biomass constraints from 0 to z_growth*
      4. At each point, maximise product (with biomass ≥ constraint)
      5. Return (growth_values, product_values)

    Raises
    ------
    ModelNotFoundError          → HTTP 404
    ReactionNotFoundError        → HTTP 404  (bad biomass or product ID)
    InfeasibleModelError         → HTTP 422  (biomass LP infeasible)
    ComputationTooExpensiveError  → HTTP 400  (large model + many steps)
    """
    model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    n_reactions = len(model.reactions)

    # Computation cost guardrail
    if n_reactions > _LARGE_MODEL_REACTION_LIMIT and request.steps > _LARGE_MODEL_STEPS_LIMIT:
        raise ComputationTooExpensiveError(
            f"Model has {n_reactions:,} reactions and {request.steps} steps were requested. "
            f"For models > {_LARGE_MODEL_REACTION_LIMIT:,} reactions limit steps to "
            f"≤ {_LARGE_MODEL_STEPS_LIMIT} to avoid excessive compute time."
        )

    # Validate reaction IDs exist in model
    rxn_ids = {r.id for r in model.reactions}
    if request.product_reaction not in rxn_ids:
        raise ReactionNotFoundError(request.product_reaction)
    if request.biomass_reaction not in rxn_ids:
        raise ReactionNotFoundError(request.biomass_reaction)

    with model:
        configure_solver_in_context(model, request.solver)
        apply_medium_from_metadata(model, model_id)  # ← medium overlay

        biomass_rxn = model.reactions.get_by_id(request.biomass_reaction)
        product_rxn = model.reactions.get_by_id(request.product_reaction)

        # ── Step 1: maximise biomass to determine scan range ──────────
        model.objective = biomass_rxn
        model.objective.direction = "max"
        baseline = model.optimize()
        if baseline.status != "optimal":
            raise InfeasibleModelError(
                f"Biomass reaction '{request.biomass_reaction}' is infeasible: "
                f"solver status = {baseline.status}."
            )
        max_growth = float(baseline.objective_value)
        solver_str = _solver_name(model, request.solver)

        # ── Step 2: switch objective to production ─────────────────────
        model.objective = product_rxn
        model.objective.direction = "max"

        # ── Step 3: scan biomass constraints ───────────────────────────
        # linspace(0, max_growth, steps) samples the full growth range.
        # At growth = 0 the cell is not growing → maximum product possible.
        # At growth = max_growth the cell achieves full optima → product
        # flux may be limited (trade-off).
        growth_points: List[float] = list(np.linspace(0.0, max_growth, request.steps))
        product_values: List[float] = []

        original_lb = biomass_rxn.lower_bound  # remember original for logging
        for g in growth_points:
            biomass_rxn.lower_bound = g         # tracked by COBRApy context
            sol = model.optimize()
            if sol.status == "optimal":
                product_values.append(float(sol.objective_value))
            else:
                # Infeasible at this growth constraint → record 0
                # (can happen near max_growth due to numeric rounding)
                product_values.append(0.0)

    # Context exits here → biomass_rxn.lower_bound restored to original_lb
    # Context exits here → model.objective restored to original
    max_product = max(product_values) if product_values else 0.0

    logger.info(
        "Envelope computed: model_id=%s biomass=%s product=%s steps=%d "
        "max_growth=%.4f max_product=%.4f",
        model_id, request.biomass_reaction, request.product_reaction,
        request.steps, max_growth, max_product,
    )

    return ProductionEnvelopeResponse(
        success=True,
        message=(
            f"Production envelope computed: {request.steps} points, "
            f"max growth = {max_growth:.4f}, max product = {max_product:.4f}."
        ),
        model_id=model_id,
        biomass_reaction=request.biomass_reaction,
        product_reaction=request.product_reaction,
        max_growth=max_growth,
        max_product=max_product,
        steps=request.steps,
        growth_values=growth_points,
        product_values=product_values,
        solver_status=baseline.status,
        solver_name=solver_str,
    )


# ─────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────


def _solver_name(model: cobra.Model, fallback: str) -> str:
    """Extract a human-readable solver name from the model's interface."""
    try:
        return model.solver.interface.__name__.split(".")[-2]
    except Exception:
        return fallback
