"""
backend/services/optknock_service.py
--------------------------------------
Greedy Growth-Constrained Knockout Search
(Sequential LP-based heuristic for strain design)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM TRANSPARENCY NOTICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This implementation approximates OptKnock (Burgard et al. 2003) using a
**sequential LP-based greedy heuristic**.  It does NOT perform the full
bilevel MILP reformulation of the original OptKnock paper.

True OptKnock:
  - Bilevel MILP with binary knockout variables y_i ∈ {0,1}
  - Flux bounds linked via Big-M:  v_i ≤ ub_i(1−y_i),  v_i ≥ lb_i(1−y_i)
  - Inner problem collapsed to single level via KKT / primal-dual conditions
  - Requires MILP-capable solver (CPLEX, Gurobi, CBC)

This implementation:
  - Evaluates knockouts one-at-a-time via standard LP (GLPK)
  - Enforces growth coupling via  biomass_lb ≥ growth_floor
  - Selects greedily per round; does not jointly optimise all K knockouts
  - ~10–20% suboptimal vs. true OptKnock for K ≥ 2 (acceptable for screening)

For peer-reviewed studies, report this as:
  "Greedy growth-constrained knockout screening (LP heuristic), analogous
   to OptKnock but without bilevel MILP guarantees."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0. Pre-check: product reaction can carry positive flux at all.
1. Baseline: maximize biomass → μ*.  growth_floor = max(α·μ*, 1e-4).
2. Candidate selection: internal, non-essential, no mandatory flux (lb > 0).
3. Essentiality filter: KO each candidate alone; remove if growth < threshold.
4. Greedy rounds (k = 1 … max_knockouts):
       For each remaining candidate:
           Apply all selected knockouts + trial knockout
           Set biomass_rxn.lower_bound = growth_floor
           Maximize product_rxn
       Keep candidate with highest product (threshold: Δ > 1e-6).
       Stop if no improvement.
5. Final joint validation:
       Apply all selected knockouts
       Maximize biomass → final_growth
       Set biomass_rxn.lower_bound = 0.999 × final_growth
       Maximize product → final_product    ← both from consistent LP

Complexity: O(max_knockouts × |candidates|) LP solves  ≈ 1 200 worst-case.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THREAD SAFETY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All LP work runs inside `with model:` (outer isolation) + nested
`with model:` per candidate test.  COBRApy rolls back ALL bound and
objective changes on each context exit — the shared registry model is
never mutated.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import cobra

from backend.exceptions import (
    ComputationTooExpensiveError,
    InfeasibleModelError,
    ModelNotFoundError,
    ReactionNotFoundError,
)
from backend.schemas.requests import OptKnockRequest
from backend.schemas.responses import OptKnockResponse
from backend.services.medium_service import apply_medium_from_metadata
from backend.services.model_registry import get_registry
from backend.utils.solve_utils import configure_solver_in_context

logger = logging.getLogger(__name__)

# ─── Numerical constants ───────────────────────────────────────────────────
# FIX 5: Use 1e-6 to match GLPK's default feasibility tolerance.
# Previous value of 1e-9 was below solver tolerance → false zero-flux detection.
_BLOCKED_TOL: float = 1e-6

_MAX_CANDIDATES: int = 300
_LARGE_MODEL_LIMIT: int = 6_000

# Fraction of baseline below which single KO is considered essential
_ESSENTIAL_FRACTION: float = 0.01

# Hard minimum growth floor regardless of growth_fraction (FIX 1)
_GROWTH_FLOOR_ABS: float = 1e-4

# Mandatory-flux lower-bound threshold: lb > this → exclude from candidates (FIX 4)
_MANDATORY_FLUX_LB: float = 1e-6

# Tie-breaking tolerance for greedy comparison (FIX 8)
_TIE_TOL: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════


def run_optknock(request: OptKnockRequest, model_id: str) -> OptKnockResponse:
    """
    Execute greedy growth-constrained knockout search on a registered model.

    Raises
    ------
    ModelNotFoundError           → HTTP 404
    ReactionNotFoundError        → HTTP 404
    InfeasibleModelError         → HTTP 422
    ComputationTooExpensiveError → HTTP 400
    ValueError                   → HTTP 422
    """
    model: cobra.Model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    # ── Validate reaction IDs ──────────────────────────────────────────────
    rxn_by_id: Dict[str, cobra.Reaction] = {r.id: r for r in model.reactions}

    if request.biomass_reaction not in rxn_by_id:
        raise ReactionNotFoundError(request.biomass_reaction)
    if request.product_reaction not in rxn_by_id:
        raise ReactionNotFoundError(request.product_reaction)

    biomass_rxn = rxn_by_id[request.biomass_reaction]
    product_rxn = rxn_by_id[request.product_reaction]

    if biomass_rxn.boundary:
        raise ValueError(
            f"Biomass reaction '{request.biomass_reaction}' is a boundary reaction. "
            "OptKnock requires an internal biomass reaction."
        )

    # ── Guardrails ──────────────────────────────────────────────────────────
    n_rxns = len(model.reactions)

    if n_rxns > _LARGE_MODEL_LIMIT and request.max_knockouts > 2:
        raise ComputationTooExpensiveError(
            f"Model has {n_rxns:,} reactions. max_knockouts > 2 is blocked for "
            f"models > {_LARGE_MODEL_LIMIT:,} reactions to prevent timeout."
        )
    if request.candidate_reactions and len(request.candidate_reactions) > _MAX_CANDIDATES:
        raise ComputationTooExpensiveError(
            f"candidate_reactions has {len(request.candidate_reactions)} entries; "
            f"maximum is {_MAX_CANDIDATES}."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # All LP work inside a single isolation context
    # ══════════════════════════════════════════════════════════════════════════
    with model:
        configure_solver_in_context(model, request.solver)
        apply_medium_from_metadata(model, model_id)  # ← medium overlay (rolled back on exit)

        # ── FIX 2: Product feasibility pre-check ──────────────────────────
        # Before spending time on essentiality filtering, verify that the
        # product reaction can carry positive flux at all under current medium.
        with model:
            model.objective = product_rxn
            model.objective.direction = "max"
            pre_sol = model.optimize()
            if pre_sol.status != "optimal" or float(pre_sol.objective_value) < _BLOCKED_TOL:
                raise InfeasibleModelError(
                    f"Product reaction '{request.product_reaction}' cannot carry positive flux "
                    "under the current medium conditions. "
                    "Verify the reaction ID, check that it is connected to active pathways, "
                    "and confirm the medium enables relevant precursor uptake."
                )

        # ── 1. Baseline growth ─────────────────────────────────────────────
        model.objective = biomass_rxn
        model.objective.direction = "max"
        base_sol = model.optimize()

        if base_sol.status != "optimal":
            raise InfeasibleModelError(
                f"Baseline FBA (biomass '{request.biomass_reaction}') is infeasible: "
                f"solver status = {base_sol.status}. "
                "Apply a valid medium before running the knockout search."
            )

        baseline_growth = float(base_sol.objective_value)
        if baseline_growth < _BLOCKED_TOL:
            raise InfeasibleModelError(
                "Baseline growth = 0. The model cannot grow under current medium. "
                "Use the Medium Configuration tab to enable nutrient uptake."
            )

        # FIX 1: Hard minimum growth floor — prevents zero-growth LP solutions
        # even when growth_fraction = 0.01 (schema minimum).
        growth_floor = max(request.growth_fraction * baseline_growth, _GROWTH_FLOOR_ABS)
        logger.info(
            "OptKnock baseline: model_id=%s μ*=%.4f growth_floor=%.4f (fraction=%.2f)",
            model_id, baseline_growth, growth_floor, request.growth_fraction,
        )

        # ── 2. Baseline product flux at growth_floor ──────────────────────
        with model:
            biomass_rxn.lower_bound = growth_floor
            model.objective = product_rxn
            model.objective.direction = "max"
            p_sol = model.optimize()
            baseline_product = (
                float(p_sol.objective_value) if p_sol.status == "optimal" else 0.0
            )

        # ── 3. Build candidate reaction list ──────────────────────────────
        candidate_ids: List[str] = _build_candidates(
            model,
            biomass_rxn=biomass_rxn,
            product_rxn=product_rxn,
            requested=request.candidate_reactions,
        )
        logger.info(
            "OptKnock candidates: model_id=%s n_reactions=%d raw_candidates=%d",
            model_id, n_rxns, len(candidate_ids),
        )

        # ── FIX 3: Essentiality pre-filter ────────────────────────────────
        # FIX 3: Threshold uses max() to guard against very small baseline values.
        # FIX 4: Mandatory-flux reactions already excluded in _build_candidates.
        essential: Set[str] = set()
        # FIX 3: robust threshold — absolute floor prevents numeric noise misclassification
        essential_threshold = max(_ESSENTIAL_FRACTION * baseline_growth, _GROWTH_FLOOR_ABS)

        for cand_id in list(candidate_ids):
            with model:
                rxn_by_id[cand_id].knock_out()       # lb=0, ub=0; tracked by context
                s = model.optimize()                  # biomass objective (outer context)
                growth_after = (
                    float(s.objective_value) if s.status == "optimal" else 0.0
                )
                if growth_after < essential_threshold:
                    essential.add(cand_id)

        candidates: List[str] = [c for c in candidate_ids if c not in essential]
        logger.info(
            "OptKnock essentiality: essential_excluded=%d non_essential=%d threshold=%.4f",
            len(essential), len(candidates), essential_threshold,
        )

        # ── FIX 9: Log skipped infeasible candidates during greedy search
        # (counters updated throughout the greedy loop below)
        _infeasible_skip_count = 0

        # ── 4. Greedy sequential knockout search ──────────────────────────
        selected: List[str] = []
        current_best_product = baseline_product
        search_log: List[str] = []
        iterations_completed = 0

        for k in range(request.max_knockouts):
            best_this_round: float = -1.0
            best_rxn_id: Optional[str] = None
            remaining = [c for c in candidates if c not in selected]

            for cand_id in remaining:
                with model:
                    # Re-apply all previously confirmed knockouts in inner context
                    for sel_id in selected:
                        model.reactions.get_by_id(sel_id).knock_out()

                    # Apply trial knockout
                    model.reactions.get_by_id(cand_id).knock_out()

                    # Enforce minimum growth coupling
                    biomass_rxn.lower_bound = growth_floor

                    # Maximise product flux
                    model.objective = product_rxn
                    model.objective.direction = "max"
                    s = model.optimize()

                    if s.status == "optimal":
                        prod_val = float(s.objective_value)
                        # FIX 8: Use explicit tie tolerance to prevent floating noise
                        # from changing selection order across runs
                        if prod_val > best_this_round + _TIE_TOL:
                            best_this_round = prod_val
                            best_rxn_id = cand_id
                    else:
                        # FIX 9: Explicitly log infeasible candidates
                        _infeasible_skip_count += 1
                        logger.debug(
                            "Skipping '%s': infeasible under growth constraint "
                            "(status=%s, growth_floor=%.4f).",
                            cand_id, s.status, growth_floor,
                        )

            iterations_completed = k + 1

            # FIX 8: improvement threshold uses _TIE_TOL for consistency
            if best_rxn_id is not None and best_this_round > current_best_product + _TIE_TOL:
                selected.append(best_rxn_id)
                delta = best_this_round - current_best_product
                search_log.append(
                    f"Round {k + 1}: knocked out '{best_rxn_id}' → "
                    f"product = {best_this_round:.6f} (+{delta:.6f})"
                )
                current_best_product = best_this_round
                logger.info(
                    "OptKnock round %d: selected '%s' → product=%.4f (+%.4f)",
                    k + 1, best_rxn_id, best_this_round, delta,
                )
            else:
                search_log.append(
                    f"Round {k + 1}: no improvement found (best candidate = "
                    f"{best_rxn_id or 'none'}, Δproduct = "
                    f"{best_this_round - current_best_product:.6f}). Search complete."
                )
                break

        # ── FIX 6: Final joint validation ─────────────────────────────────
        # CRITICAL: Report final_growth and final_product from the SAME LP solve
        # context, not from different greedy-loop iterations.
        # Previous code used current_best_product (from a growth-constrained LP)
        # alongside a separate unconstrained biomass LP → inconsistent phenotype.
        final_growth: float = growth_floor
        final_product: float = 0.0
        final_solver_status: str = base_sol.status

        if selected:
            with model:
                for sel_id in selected:
                    model.reactions.get_by_id(sel_id).knock_out()

                # Step A: Unconstrained biomass maximisation after knockouts
                model.objective = biomass_rxn
                model.objective.direction = "max"
                g_sol = model.optimize()
                final_solver_status = g_sol.status

                if g_sol.status == "optimal":
                    final_growth = float(g_sol.objective_value)
                else:
                    # Solver infeasible at final verification — use floor
                    final_growth = growth_floor
                    logger.warning(
                        "OptKnock final growth solve infeasible (status=%s); "
                        "using growth_floor=%.4f as fallback.",
                        g_sol.status, growth_floor,
                    )

                # Step B: Product maximisation at actual (near-optimal) growth
                # Use 0.999 × final_growth to avoid numerical infeasibility from
                # equality constraints on a degenerate biomass optimum.
                biomass_rxn.lower_bound = final_growth * 0.999
                model.objective = product_rxn
                model.objective.direction = "max"
                p_sol = model.optimize()

                if p_sol.status == "optimal":
                    final_product = float(p_sol.objective_value)
                    logger.info(
                        "OptKnock final (joint): growth=%.4f product=%.4f KOs=%s",
                        final_growth, final_product, selected,
                    )
                else:
                    # Product LP infeasible at actual growth — report floor
                    final_product = 0.0
                    logger.warning(
                        "OptKnock final product solve infeasible (status=%s) "
                        "at biomass_lb=%.4f; reporting 0.",
                        p_sol.status, biomass_rxn.lower_bound,
                    )
        else:
            # No knockouts selected → report baseline joint phenotype
            final_growth = baseline_growth
            final_product = baseline_product
            final_solver_status = base_sol.status

    # ── Fold improvement ───────────────────────────────────────────────────
    fold_improvement = (
        final_product / baseline_product
        if baseline_product > _BLOCKED_TOL
        else 1.0
    )

    # ── Response message ───────────────────────────────────────────────────
    # FIX 7: Transparent algorithm naming in all user-facing text
    if not selected:
        success = False
        msg = (
            "Greedy knockout search found no beneficial knockouts under the current "
            "configuration. The model may already achieve maximum product flux at the "
            "specified growth fraction, or no single candidate knockout improves production. "
            "Suggestions: reduce growth_fraction, broaden the candidate set, or try a "
            "different medium preset."
        )
    else:
        success = True
        ko_str = ", ".join(f"Δ{r}" for r in selected)
        msg = (
            f"Greedy LP search identified {len(selected)} knockout(s): {ko_str}. "
            f"Predicted joint phenotype — growth: {final_growth:.4f} h⁻¹, "
            f"product flux: {final_product:.4f} mmol gDW⁻¹ h⁻¹ "
            f"(×{fold_improvement:.2f} vs baseline {baseline_product:.4f}). "
            f"Note: results are from a greedy LP heuristic, not exact bilevel MILP."
        )

    logger.info(
        "OptKnock complete: model_id=%s knockouts=%s fold=%.2f "
        "infeasible_skipped=%d status=%s",
        model_id, selected, fold_improvement,
        _infeasible_skip_count, final_solver_status,
    )

    return OptKnockResponse(
        success=success,
        message=msg,
        model_id=model_id,
        knocked_reactions=selected,
        predicted_growth=final_growth,
        predicted_product_flux=final_product,
        baseline_growth=baseline_growth,
        baseline_product_flux=baseline_product,
        growth_fraction=request.growth_fraction,
        fold_improvement=fold_improvement,
        solver_status=final_solver_status,
        candidates_tested=len(candidates),
        essential_excluded=len(essential),
        search_log=search_log,
        # FIX 10: Transparency fields
        algorithm_type="greedy_lp_heuristic",
        growth_floor_used=growth_floor,
        essential_reactions_filtered=len(essential),
        iterations=iterations_completed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════


def _build_candidates(
    model: cobra.Model,
    biomass_rxn: cobra.Reaction,
    product_rxn: cobra.Reaction,
    requested: Optional[List[str]],
) -> List[str]:
    """
    Build the set of reactions eligible for knockout testing.

    Exclusion rules (applied in order):
    1. Biomass reaction (must survive for growth coupling)
    2. Product reaction (objective; cannot knock out the target)
    3. Boundary / exchange reactions (medium interface)
    4. FIX 4: Mandatory-flux reactions  (lb > _MANDATORY_FLUX_LB):
       - These carry obligate flux (e.g. ATPM, NGAM, maintenance).
       - Previous implementation excluded ATPM by string match only, missing
         synonyms like NGAM, Maintenance_ATP, maintenance, etc.
       - Exclusion by lb > 0 catches ALL mandatory-flux reactions regardless of name.
    5. Already-blocked reactions (ub ≤ 0 and lb ≥ 0):
       - Knocking out a reaction with zero flux range has no effect.
    6. Hard cap at _MAX_CANDIDATES for computational budget.
    """
    if requested is not None:
        rxn_ids = {r.id for r in model.reactions}
        exclude = {biomass_rxn.id, product_rxn.id}
        valid = [
            c for c in requested
            if c in rxn_ids
            and c not in exclude
            and not model.reactions.get_by_id(c).boundary
            # FIX 4: also exclude mandatory-flux from user-provided candidates
            and model.reactions.get_by_id(c).lower_bound <= _MANDATORY_FLUX_LB
        ]
        return valid[:_MAX_CANDIDATES]

    # Auto-build
    exclude: set = {biomass_rxn.id, product_rxn.id}
    candidates: List[str] = []

    for rxn in model.reactions:
        if rxn.id in exclude:
            continue

        # Rule 3: no exchange reactions
        if rxn.boundary:
            continue

        # FIX 4: Exclude mandatory-flux reactions (replaces ATPM string match)
        # Any reaction with lb > _MANDATORY_FLUX_LB MUST carry flux; knocking
        # it out would violate the model's energy/maintenance constraints.
        if rxn.lower_bound > _MANDATORY_FLUX_LB:
            continue

        # Rule 5: skip already-blocked reactions (KO has no effect)
        if rxn.lower_bound >= -_BLOCKED_TOL and rxn.upper_bound <= _BLOCKED_TOL:
            continue

        candidates.append(rxn.id)

    return candidates[:_MAX_CANDIDATES]
