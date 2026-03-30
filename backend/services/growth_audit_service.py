"""
backend/services/growth_audit_service.py
------------------------------------------
Scientific debugging service: diagnoses why biomass growth = 0 despite
solver status = optimal.

This is a diagnostic tool only — it NEVER permanently mutates the shared
model.  All LP work happens inside `with model:` contexts.

Audit structure
---------------
The function runs 10 ordered checks and collects findings into a structured
`GrowthAuditResponse`.  Finally it applies a decision-tree inference to
produce a `likely_cause` string — a single human-readable explanation of
the most probable root cause.

Decision tree (in priority order)
-----------------------------------
1. growth > 0         → no issue
2. uptake_count == 0  → all exchanges closed  (most common user error)
3. no carbon enabled  → no carbon source
4. no nitrogen        → no nitrogen source
5. no phosphate       → no phosphate
6. structurally blocked biomass → missing precursor pathway
7. ATPM > 0 and growth = 0 → ATPM may over-consume ATP
8. objective is an exchange or has no metabolites → wrong biomass reaction
9. fallback           → unknown (suggests stoichiometric error)

Thread safety
-------------
- Read-only checks run before/after `with model:` (no risk).
- LP solves run inside `with model:` — solver config rolled back on exit.
- The shared registry model is left pristine after each call.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cobra
import cobra.flux_analysis

from backend.exceptions import ModelNotFoundError
from backend.schemas.responses import (
    ATPMInfo,
    BiomassAuditInfo,
    ExchangeStatus,
    GrowthAuditResponse,
)
from backend.services.model_registry import get_registry
from backend.utils.solve_utils import configure_solver_in_context
from backend.services.medium_service import apply_medium_from_metadata

logger = logging.getLogger(__name__)

# ─── Canonical exchange reaction IDs (BiGG namespace) ─────────────────────
_CARBON_IDS: List[str] = [
    "EX_glc__D_e",  # D-Glucose (most common)
    "EX_ac_e",      # Acetate
    "EX_succ_e",    # Succinate
    "EX_lac__D_e",  # D-Lactate
    "EX_fru_e",     # Fructose
    "EX_glyc_e",    # Glycerol
]
_NITROGEN_IDS: List[str] = [
    "EX_nh4_e",     # Ammonium (most common)
    "EX_no3_e",     # Nitrate
    "EX_gln__L_e",  # L-Glutamine (organic N)
]
_PHOSPHATE_SULFATE_IDS: List[str] = [
    "EX_pi_e",      # Inorganic phosphate
    "EX_so4_e",     # Sulfate
]
_OXYGEN_ID = "EX_o2_e"
_BLOCKED_TOL = 1e-9


def run_growth_diagnostic(model_id: str) -> GrowthAuditResponse:
    """
    Execute the full growth-zero diagnostic suite on a registered model.

    Returns a fully populated `GrowthAuditResponse` with:
    - Per-nutrient exchange status
    - Baseline solve result
    - ATPM constraint inspection
    - Structural blockage check (single-reaction FVA)
    - `likely_cause`: a single inferred root-cause string

    Raises
    ------
    ModelNotFoundError
        Route catches this → HTTP 404.
    """
    model: cobra.Model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    audit: List[str] = []   # step-by-step human-readable log

    # ══════════════════════════════════════════════════════════════════
    # 1. Identify the biomass / objective reaction (read-only)
    # ══════════════════════════════════════════════════════════════════
    obj_rxns = [r for r in model.reactions if abs(r.objective_coefficient) > _BLOCKED_TOL]

    if not obj_rxns:
        audit.append("❌ No objective reaction found in model.")
        return GrowthAuditResponse(
            success=False,
            message="No objective reaction is set. Use POST /models/{id}/objective to assign one.",
            model_id=model_id,
            likely_cause=(
                "Model has no objective function. "
                "Set one via POST /models/{id}/objective before running FBA."
            ),
            audit_log=audit,
        )

    biomass_rxn = obj_rxns[0]
    biomass_info = BiomassAuditInfo(
        reaction_id=biomass_rxn.id,
        lower_bound=biomass_rxn.lower_bound,
        upper_bound=biomass_rxn.upper_bound,
        num_metabolites=len(biomass_rxn.metabolites),
        is_exchange=biomass_rxn.boundary,
        has_metabolites=len(biomass_rxn.metabolites) > 0,
    )
    audit.append(
        f"✅ Biomass reaction: {biomass_rxn.id} "
        f"(lb={biomass_rxn.lower_bound}, ub={biomass_rxn.upper_bound}, "
        f"metabolites={biomass_info.num_metabolites})"
    )
    if biomass_info.is_exchange:
        audit.append(
            "⚠️  Objective is a boundary/exchange reaction — "
            "biomass reactions are typically internal."
        )
    if not biomass_info.has_metabolites:
        audit.append("❌ Objective reaction has NO metabolites — likely the wrong reaction.")

    # ══════════════════════════════════════════════════════════════════
    # 2–6. Exchange reactions — read-only bound inspection
    # ══════════════════════════════════════════════════════════════════
    rxn_by_id: Dict[str, cobra.Reaction] = {r.id: r for r in model.reactions}

    carbon_sources = _check_exchanges(_CARBON_IDS, rxn_by_id, audit, "Carbon")
    nitrogen_sources = _check_exchanges(_NITROGEN_IDS, rxn_by_id, audit, "Nitrogen")
    phosphate_sulfate = _check_exchanges(_PHOSPHATE_SULFATE_IDS, rxn_by_id, audit, "Phosphate/Sulfate")
    oxygen_status = _check_single_exchange(_OXYGEN_ID, rxn_by_id, audit, "Oxygen")

    # ══════════════════════════════════════════════════════════════════
    # 7. ATPM / maintenance reaction inspection (read-only)
    # ══════════════════════════════════════════════════════════════════
    atpm_rxn_candidates = [r for r in model.reactions if "ATPM" in r.id.upper()]
    if atpm_rxn_candidates:
        atpm_rxn = atpm_rxn_candidates[0]
        atpm_status = ATPMInfo(
            found=True,
            reaction_id=atpm_rxn.id,
            lower_bound=atpm_rxn.lower_bound,
            upper_bound=atpm_rxn.upper_bound,
            maintenance_required=atpm_rxn.lower_bound > _BLOCKED_TOL,
        )
        audit.append(
            f"{'⚠️' if atpm_status.maintenance_required else '✅'} "
            f"ATPM: {atpm_rxn.id} "
            f"lb={atpm_rxn.lower_bound:.4f}, ub={atpm_rxn.upper_bound:.4f}"
            + (
                f" — maintenance demand active ({atpm_rxn.lower_bound:.2f} mmol gDW⁻¹ h⁻¹)"
                if atpm_status.maintenance_required else ""
            )
        )
    else:
        atpm_status = ATPMInfo(found=False)
        audit.append("ℹ️  No ATPM reaction found (not necessarily an error).")

    # ══════════════════════════════════════════════════════════════════
    # 8. Exchange uptake count (read-only)
    # ══════════════════════════════════════════════════════════════════
    exchange_rxns = [r for r in model.reactions if r.boundary]
    exchange_uptake_count = sum(1 for r in exchange_rxns if r.lower_bound < 0)
    total_exchanges = len(exchange_rxns)
    audit.append(
        f"{'⚠️' if exchange_uptake_count == 0 else '✅'} "
        f"Exchange uptake-enabled reactions: {exchange_uptake_count} / {total_exchanges}"
    )

    # ══════════════════════════════════════════════════════════════════
    # 9. Baseline FBA solve + structural FVA — inside `with model:`
    # ══════════════════════════════════════════════════════════════════
    growth_rate = 0.0
    solver_status = "not_run"
    structurally_blocked: Optional[bool] = None

    with model:
        configure_solver_in_context(model, "glpk")
        apply_medium_from_metadata(model, model_id)  # ← medium overlay

        # ── Baseline solve ────────────────────────────────────────────
        solution = model.optimize()
        solver_status = solution.status
        growth_rate = (
            float(solution.objective_value)
            if solver_status == "optimal"
            else 0.0
        )
        audit.append(
            f"{'✅' if growth_rate > _BLOCKED_TOL else '❌'} "
            f"Baseline FBA: status={solver_status}, "
            f"growth_rate={growth_rate:.8f}"
        )

        # ── Single-reaction FVA: is biomass structurally blocked? ─────
        # Run FVA at fraction=0 so we check the widest possible range
        # (not just what's optimal — we want to know if ANY flux through
        # biomass is feasible at all).
        try:
            fva = cobra.flux_analysis.flux_variability_analysis(
                model,
                reaction_list=[biomass_rxn],
                fraction_of_optimum=0.0,
                processes=1,
            )
            fva_min = float(fva.loc[biomass_rxn.id, "minimum"])
            fva_max = float(fva.loc[biomass_rxn.id, "maximum"])
            structurally_blocked = (
                abs(fva_min) < _BLOCKED_TOL and abs(fva_max) < _BLOCKED_TOL
            )
            audit.append(
                f"{'❌' if structurally_blocked else '✅'} "
                f"Biomass FVA (fraction=0): "
                f"min={fva_min:.8f}, max={fva_max:.8f} "
                f"→ {'BLOCKED' if structurally_blocked else 'can carry flux'}"
            )
        except Exception as exc:
            audit.append(f"⚠️  Biomass structural FVA skipped: {exc}")
            logger.warning("Biomass FVA in growth audit failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════
    # 10. Infer the most likely root cause
    # ══════════════════════════════════════════════════════════════════
    likely_cause = _infer_likely_cause(
        growth_rate=growth_rate,
        solver_status=solver_status,
        carbon_sources=carbon_sources,
        nitrogen_sources=nitrogen_sources,
        phosphate_sulfate=phosphate_sulfate,
        oxygen_status=oxygen_status,
        atpm=atpm_status,
        exchange_uptake_count=exchange_uptake_count,
        structurally_blocked=structurally_blocked,
        biomass_info=biomass_info,
    )
    audit.append(f"🔍 Likely cause: {likely_cause}")

    logger.info(
        "Growth audit complete: model_id=%s growth=%.6f status=%s cause=%r",
        model_id, growth_rate, solver_status, likely_cause,
    )

    return GrowthAuditResponse(
        success=True,
        message=f"Growth audit complete. {likely_cause}",
        model_id=model_id,
        biomass_reaction=biomass_info,
        growth_rate=growth_rate,
        solver_status=solver_status,
        carbon_sources=carbon_sources,
        nitrogen_sources=nitrogen_sources,
        phosphate_sulfate=phosphate_sulfate,
        oxygen=oxygen_status,
        atpm_bounds=atpm_status,
        exchange_uptake_count=exchange_uptake_count,
        total_exchanges=total_exchanges,
        biomass_structurally_blocked=structurally_blocked,
        likely_cause=likely_cause,
        audit_log=audit,
    )


# ─── Private helpers ───────────────────────────────────────────────────────


def _check_exchanges(
    ids: List[str],
    rxn_by_id: Dict[str, cobra.Reaction],
    audit: List[str],
    label: str,
) -> Dict[str, ExchangeStatus]:
    """
    Check a list of canonical exchange reaction IDs.
    Returns a dict of {reaction_id: ExchangeStatus} for reactions present in the model.
    Reactions absent from the model are silently skipped (logged in audit).
    """
    result: Dict[str, ExchangeStatus] = {}
    any_enabled = False
    for rxn_id in ids:
        rxn = rxn_by_id.get(rxn_id)
        if rxn is None:
            continue   # absent from this model — not necessarily a problem
        enabled = rxn.lower_bound < 0
        any_enabled = any_enabled or enabled
        result[rxn_id] = ExchangeStatus(
            reaction_id=rxn_id,
            lower_bound=rxn.lower_bound,
            upper_bound=rxn.upper_bound,
            uptake_enabled=enabled,
        )
        audit.append(
            f"  {'✅' if enabled else '⛔'} {label} {rxn_id}: "
            f"lb={rxn.lower_bound}, ub={rxn.upper_bound} "
            f"({'uptake ON' if enabled else 'uptake OFF — closed'})"
        )
    if result and not any_enabled:
        audit.append(f"  ⚠️  All known {label} exchanges are closed (lb ≥ 0).")
    return result


def _check_single_exchange(
    rxn_id: str,
    rxn_by_id: Dict[str, cobra.Reaction],
    audit: List[str],
    label: str,
) -> Optional[ExchangeStatus]:
    """Check one specific exchange reaction. Returns None if not present."""
    rxn = rxn_by_id.get(rxn_id)
    if rxn is None:
        audit.append(f"  ℹ️  {label} ({rxn_id}): not present in model.")
        return None
    enabled = rxn.lower_bound < 0
    status = ExchangeStatus(
        reaction_id=rxn_id,
        lower_bound=rxn.lower_bound,
        upper_bound=rxn.upper_bound,
        uptake_enabled=enabled,
    )
    audit.append(
        f"  {'✅' if enabled else '💤'} {label} ({rxn_id}): "
        f"lb={rxn.lower_bound}, ub={rxn.upper_bound} "
        f"({'uptake ON' if enabled else 'uptake OFF — anaerobic?'})"
    )
    return status


def _infer_likely_cause(
    growth_rate: float,
    solver_status: str,
    carbon_sources: Dict,
    nitrogen_sources: Dict,
    phosphate_sulfate: Dict,
    oxygen_status: Optional[ExchangeStatus],
    atpm: ATPMInfo,
    exchange_uptake_count: int,
    structurally_blocked: Optional[bool],
    biomass_info: BiomassAuditInfo,
) -> str:
    """
    Decision-tree inference of the most likely root cause of zero growth.
    Returns a single descriptive sentence for the `likely_cause` field.
    """
    # ── Case 0: growth is already positive ──────────────────────────
    if growth_rate > _BLOCKED_TOL:
        return (
            f"Growth is positive ({growth_rate:.6f}) — no blockage detected. "
            "If the value seems low, check medium composition or ATPM constraints."
        )

    # ── Case 1: solver failed ────────────────────────────────────────
    if solver_status != "optimal":
        return (
            f"LP solver returned '{solver_status}' (not optimal). "
            "The model may be infeasible due to conflicting constraints — "
            "check for lb > ub violations using the /validation endpoint."
        )

    # ── Case 2: no nutrient uptake at all ───────────────────────────
    if exchange_uptake_count == 0:
        return (
            "All exchange reactions are closed (lb ≥ 0 everywhere). "
            "The cell has no nutrient supply. Open at least one carbon, "
            "one nitrogen, and one phosphate exchange to lb < 0."
        )

    # ── Case 3: biomass reaction itself is wrong ─────────────────────
    if not biomass_info.has_metabolites:
        return (
            f"The objective reaction '{biomass_info.reaction_id}' has no metabolites. "
            "This is likely not a valid biomass reaction. "
            "Confirm the correct reaction ID and set it via POST /models/{id}/objective."
        )
    if biomass_info.is_exchange:
        return (
            f"The objective '{biomass_info.reaction_id}' is a boundary (exchange) reaction. "
            "The biomass reaction should be an internal reaction. "
            "Check and reset the objective."
        )

    # ── Case 4: no carbon source enabled ────────────────────────────
    any_carbon = any(s.uptake_enabled for s in carbon_sources.values())
    if not any_carbon and carbon_sources:
        return (
            "No common carbon source has uptake enabled (all lb ≥ 0). "
            "Enable glucose or another carbon exchange: set lb < 0 on EX_glc__D_e "
            "or equivalent."
        )

    # ── Case 5: no nitrogen source enabled ──────────────────────────
    any_nitrogen = any(s.uptake_enabled for s in nitrogen_sources.values())
    if not any_nitrogen and nitrogen_sources:
        return (
            "No nitrogen source has uptake enabled. "
            "Amino acids and nucleotides require nitrogen. "
            "Enable EX_nh4_e (ammonium) to lb < 0."
        )

    # ── Case 6: structurally blocked biomass ─────────────────────────
    if structurally_blocked is True:
        return (
            f"Biomass reaction '{biomass_info.reaction_id}' is structurally blocked: "
            "FVA shows max flux = 0 under all feasible conditions. "
            "One or more essential precursors (e.g. amino acids, lipids, cofactors) "
            "cannot be synthesised — check blocked reactions via /validation?run_fva=true."
        )

    # ── Case 7: ATPM over-constraining ──────────────────────────────
    if atpm.found and atpm.maintenance_required:
        return (
            f"ATPM lower bound = {atpm.lower_bound:.2f} mmol gDW⁻¹ h⁻¹. "
            "The model must satisfy this ATP maintenance demand before allocating "
            "ATP to biomass synthesis. With low carbon uptake, this may fully "
            "consume available ATP. Try reducing EX carbon uptake constraints or "
            "lowering the ATPM bound."
        )

    # ── Case 8: no phosphate/sulfate ─────────────────────────────────
    any_phos = any(s.uptake_enabled for s in phosphate_sulfate.values())
    if not any_phos and phosphate_sulfate:
        return (
            "No phosphate or sulfate source has uptake enabled. "
            "These are required for nucleotide, lipid, and coenzyme biosynthesis. "
            "Enable EX_pi_e and EX_so4_e."
        )

    # ── Case 9: oxygen missing (anaerobic hint) ───────────────────────
    if oxygen_status is not None and not oxygen_status.uptake_enabled:
        return (
            "Oxygen exchange (EX_o2_e) is closed (lb ≥ 0) — model is running in "
            "anaerobic mode. If this is correct, ensure fermentation pathways are "
            "available. If not, set EX_o2_e lb < 0 (e.g., -1000)."
        )

    # ── Fallback ──────────────────────────────────────────────────────
    return (
        "Growth is zero despite apparent nutrient availability. "
        "Possible causes: (a) missing internal reactions for a required biosynthesis step, "
        "(b) stoichiometric errors in biomass composition, "
        "(c) blocked reactions not captured by the common-metabolite checks. "
        "Run full validation with FVA enabled for a comprehensive blocked-reaction list."
    )
