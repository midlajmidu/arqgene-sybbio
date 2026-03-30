"""
backend/services/validation_service.py
----------------------------------------
Service layer for model validation checks.

Thread-safety pattern
----------------------
LP-based checks (feasibility, blocked reactions via FVA) are executed
inside a single outer `with model:` block.  Structural checks that require
no LP solve (inconsistent bounds, gene orphans, mass balance) run after
the context exits — they read model data directly without any mutation.

This guarantees the shared registry model is never polluted between requests.

Scientific purpose of each check
----------------------------------
1. Objective feasibility — LP infeasible → no growth possible under any
   exchange configuration.
2. Inconsistent bounds (lb > ub) — LP-killer; causes immediate solver error.
3. Gene-orphan reactions — internal reactions with no GPR rule; important
   for gene-deletion analysis accuracy.
4. Blocked reactions (FVA) — zero-flux under all conditions; may indicate
   missing metabolite transport.
5. Mass balance — stoichiometric element conservation per non-boundary reaction.
"""

from __future__ import annotations

import logging
from typing import Optional

import cobra

from backend.exceptions import ModelNotFoundError
from backend.schemas.requests import ValidationRequest
from backend.schemas.responses import (
    BlockedReaction,
    ErrorDetail,
    GeneOrphanReaction,
    InconsistentBound,
    MassImbalance,
    ValidationResponse,
)
from backend.services.model_registry import get_registry
from backend.utils.solve_utils import configure_solver_in_context
from backend.services.medium_service import apply_medium_from_metadata
from core.validation import (
    check_blocked_reactions,
    check_gene_orphan_reactions,
    check_inconsistent_bounds,
    check_mass_balance,
    check_objective_feasibility,
)

logger = logging.getLogger(__name__)


def run_validation(request: ValidationRequest) -> ValidationResponse:
    """
    Execute the full validation suite on a registered model.

    LP-based checks run inside a `with model:` isolation context so that
    no solver mutation persists on the shared registry instance.

    Raises
    ------
    ModelNotFoundError
        Propagates to the route layer → HTTP 404.
    """
    model: cobra.Model = get_registry().get(request.model_id)
    if model is None:
        raise ModelNotFoundError(request.model_id)

    resp = ValidationResponse(
        success=True,
        message="Validation complete.",
        model_id=request.model_id,
    )

    # ==================================================================
    # LP-based checks — run inside isolation context
    # ==================================================================
    # The outer `with model:` ensures any solver/constraint changes made
    # by individual check functions are fully rolled back when this block
    # exits.  check_objective_feasibility and check_blocked_reactions each
    # open their own inner `with model:` — nested contexts are supported.
    # ==================================================================
    with model:
        # Use glpk (always available) for validation; solver choice does
        # not affect structural correctness of these checks.
        configure_solver_in_context(model, "glpk")
        apply_medium_from_metadata(model, request.model_id)  # ← medium overlay

        # ---- 1. Feasibility ----
        try:
            resp.is_feasible, resp.feasibility_message = check_objective_feasibility(
                model
            )
            if not resp.is_feasible:
                resp.errors.append(
                    f"Infeasible objective: {resp.feasibility_message}"
                )
        except Exception as exc:
            resp.errors.append(f"Feasibility check error: {exc}")
            resp.is_feasible = False

        # ---- 4. Blocked reactions (FVA — optional, slow on large models) ----
        if request.run_fva:
            try:
                blocked_result = check_blocked_reactions(model)
                resp.blocked_count = blocked_result.count
                if (
                    blocked_result.dataframe is not None
                    and not blocked_result.dataframe.empty
                ):
                    resp.blocked_reactions = [
                        BlockedReaction(
                            reaction_id=str(row["Reaction ID"]),
                            min_flux=float(row["Min Flux"]),
                            max_flux=float(row["Max Flux"]),
                        )
                        for _, row in blocked_result.dataframe.iterrows()
                    ]
                if blocked_result.count > 0:
                    resp.warnings.append(
                        f"{blocked_result.count} blocked reaction(s) detected "
                        f"(carry no flux under any condition)."
                    )
            except Exception as exc:
                resp.warnings.append(f"Blocked reaction check skipped: {exc}")
                logger.warning("FVA blocked-reaction check failed: %s", exc)

    # ==================================================================
    # Structural checks — no LP solve, no solver mutation, safe outside context
    # ==================================================================

    # ---- 2. Inconsistent bounds ----
    try:
        bound_result = check_inconsistent_bounds(model)
        resp.inconsistent_bounds_count = bound_result.count
        if (
            bound_result.dataframe is not None
            and not bound_result.dataframe.empty
        ):
            resp.inconsistent_bounds = [
                InconsistentBound(
                    reaction_id=str(row["Reaction ID"]),
                    name=str(row.get("Name", "—")),
                    lower_bound=float(row["Lower Bound"]),
                    upper_bound=float(row["Upper Bound"]),
                )
                for _, row in bound_result.dataframe.iterrows()
            ]
        if bound_result.count > 0:
            resp.warnings.append(
                f"{bound_result.count} reaction(s) have lb > ub (infeasible bounds)."
            )
    except Exception as exc:
        resp.errors.append(f"Bound check error: {exc}")

    # ---- 3. Gene-orphan reactions ----
    try:
        orphan_result = check_gene_orphan_reactions(model)
        resp.gene_orphan_count = orphan_result.count
        if (
            orphan_result.dataframe is not None
            and not orphan_result.dataframe.empty
        ):
            resp.gene_orphan_reactions = [
                GeneOrphanReaction(
                    reaction_id=str(row["Reaction ID"]),
                    name=str(row.get("Name", "—")),
                    subsystem=str(row.get("Subsystem", "—")),
                )
                for _, row in orphan_result.dataframe.iterrows()
            ]
        if orphan_result.count > 0:
            resp.warnings.append(
                f"{orphan_result.count} reaction(s) lack gene associations (GPR)."
            )
    except Exception as exc:
        resp.errors.append(f"Gene orphan check error: {exc}")

    # ---- 5. Mass balance ----
    try:
        mb_result = check_mass_balance(model)
        resp.balanced_count = mb_result.balanced_count
        resp.unbalanced_count = mb_result.unbalanced_count
        if mb_result.dataframe is not None and not mb_result.dataframe.empty:
            resp.mass_imbalances = [
                MassImbalance(
                    reaction_id=str(row["Reaction ID"]),
                    name=str(row.get("Name", "—")),
                    imbalance=str(row["Imbalance"]),
                )
                for _, row in mb_result.dataframe.iterrows()
            ]
        if mb_result.unbalanced_count > 0:
            resp.warnings.append(
                f"{mb_result.unbalanced_count} reaction(s) are not mass-balanced."
            )
    except Exception as exc:
        resp.warnings.append(f"Mass balance check skipped: {exc}")

    # ==================================================================
    # Final status
    # ==================================================================
    if resp.errors:
        resp.success = False
        resp.message = f"Validation completed with {len(resp.errors)} error(s)."
    elif resp.warnings:
        resp.message = (
            f"Validation completed with {len(resp.warnings)} warning(s) — review below."
        )
    else:
        resp.message = "All validation checks passed. Model appears structurally sound."

    return resp
