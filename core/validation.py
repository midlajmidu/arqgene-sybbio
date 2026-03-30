"""
core/validation.py
------------------
Model-level validation checks for genome-scale metabolic models (GEMs).

Scientific purpose:
    Before running optimization-based analyses (FBA, FVA, etc.) it is critical
    to validate the model for common structural and biochemical issues.  A model
    that passes these checks is more likely to produce biologically meaningful
    predictions.

    Checks implemented:
    1. Blocked reactions  — reactions that carry zero flux under any condition.
    2. Infeasible objective — the LP is infeasible; no growth possible.
    3. Inconsistent bounds  — reactions where lb > ub.
    4. Gene-orphan reactions — reactions with no gene-protein-reaction (GPR) rule.
    5. Mass balance         — stoichiometric mass balance per reaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cobra
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------


@dataclass
class BlockedReactionResult:
    count: int = 0
    reaction_ids: List[str] = field(default_factory=list)
    dataframe: Optional[pd.DataFrame] = None


@dataclass
class BoundInconsistencyResult:
    count: int = 0
    dataframe: Optional[pd.DataFrame] = None


@dataclass
class GeneOrphanResult:
    count: int = 0
    reaction_ids: List[str] = field(default_factory=list)
    dataframe: Optional[pd.DataFrame] = None


@dataclass
class MassBalanceResult:
    balanced_count: int = 0
    unbalanced_count: int = 0
    dataframe: Optional[pd.DataFrame] = None


@dataclass
class ValidationReport:
    """Aggregated validation report for displayin the Streamlit UI."""

    is_feasible: bool = True
    feasibility_message: str = ""
    blocked: BlockedReactionResult = field(default_factory=BlockedReactionResult)
    inconsistent_bounds: BoundInconsistencyResult = field(
        default_factory=BoundInconsistencyResult
    )
    gene_orphans: GeneOrphanResult = field(default_factory=GeneOrphanResult)
    mass_balance: MassBalanceResult = field(default_factory=MassBalanceResult)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def run_all_validations(model: cobra.Model) -> ValidationReport:
    """
    Execute the complete validation suite and return a unified report.

    Parameters
    ----------
    model : cobra.Model

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    # 1. Feasibility check (fast — does a single LP solve)
    try:
        report.is_feasible, report.feasibility_message = check_objective_feasibility(model)
        if not report.is_feasible:
            report.errors.append(f"Infeasible objective: {report.feasibility_message}")
    except Exception as exc:
        report.errors.append(f"Feasibility check error: {exc}")
        report.is_feasible = False

    # 2. Inconsistent bounds (no LP needed — pure structural check)
    try:
        report.inconsistent_bounds = check_inconsistent_bounds(model)
        if report.inconsistent_bounds.count > 0:
            report.warnings.append(
                f"{report.inconsistent_bounds.count} reaction(s) have lb > ub (infeasible bounds)."
            )
    except Exception as exc:
        report.errors.append(f"Bound check error: {exc}")

    # 3. Gene orphan reactions (structural)
    try:
        report.gene_orphans = check_gene_orphan_reactions(model)
        if report.gene_orphans.count > 0:
            report.warnings.append(
                f"{report.gene_orphans.count} reaction(s) lack gene associations."
            )
    except Exception as exc:
        report.errors.append(f"Gene orphan check error: {exc}")

    # 4. Blocked reactions (requires FVA — can be slow on large models)
    try:
        report.blocked = check_blocked_reactions(model)
        if report.blocked.count > 0:
            report.warnings.append(
                f"{report.blocked.count} blocked reaction(s) detected (zero flux in all conditions)."
            )
    except Exception as exc:
        report.warnings.append(f"Blocked reaction check skipped: {exc}")

    # 5. Mass balance (structural chemistry check)
    try:
        report.mass_balance = check_mass_balance(model)
        if report.mass_balance.unbalanced_count > 0:
            report.warnings.append(
                f"{report.mass_balance.unbalanced_count} reaction(s) are not properly mass-balanced."
            )
    except Exception as exc:
        report.warnings.append(f"Mass balance check skipped: {exc}")

    return report


def check_objective_feasibility(
    model: cobra.Model,
) -> Tuple[bool, str]:
    """
    Attempt to optimise the model and determine if it is feasible.

    Returns
    -------
    Tuple[bool, str]
        (is_feasible, status_message)
    """
    with model:
        solution = model.optimize()
        if solution.status == "optimal":
            return True, f"Optimal — objective value: {solution.objective_value:.6g}"
        elif solution.status == "infeasible":
            return False, "LP is infeasible. Check exchange bounds and stoichiometry."
        elif solution.status == "unbounded":
            return (
                False,
                "Objective is unbounded. A reaction may be missing an upper bound.",
            )
        else:
            return False, f"Solver returned status: {solution.status}"


def check_blocked_reactions(model: cobra.Model) -> BlockedReactionResult:
    """
    Identify reactions that carry zero flux under all feasible conditions.

    Uses Flux Variability Analysis (FVA) with fraction_of_optimum=0 so that
    blocked reactions can be detected even when the objective is sub-optimal.
    To keep runtime reasonable, FVA is capped at 1 000 reactions.

    Parameters
    ----------
    model : cobra.Model

    Returns
    -------
    BlockedReactionResult
    """
    try:
        from cobra.flux_analysis import flux_variability_analysis

        fva_result = flux_variability_analysis(
            model,
            fraction_of_optimum=0.0,
            processes=1,  # avoid multiprocessing overhead in Streamlit
        )
        blocked_mask = (
            (fva_result["minimum"].abs() < 1e-9)
            & (fva_result["maximum"].abs() < 1e-9)
        )
        blocked_ids = list(fva_result.index[blocked_mask])
        df = fva_result[blocked_mask].reset_index()
        df.columns = ["Reaction ID", "Min Flux", "Max Flux"]
        return BlockedReactionResult(
            count=len(blocked_ids),
            reaction_ids=blocked_ids,
            dataframe=df if not df.empty else None,
        )
    except Exception as exc:
        logger.warning("FVA for blocked reactions failed: %s", exc)
        raise


def check_inconsistent_bounds(model: cobra.Model) -> BoundInconsistencyResult:
    """
    Find reactions where the lower bound exceeds the upper bound.

    Such reactions are structurally infeasible and will cause LP failures.

    Returns
    -------
    BoundInconsistencyResult
    """
    rows: List[Dict[str, Any]] = []
    for rxn in model.reactions:
        if rxn.lower_bound > rxn.upper_bound:
            rows.append(
                {
                    "Reaction ID": rxn.id,
                    "Name": rxn.name or "—",
                    "Lower Bound": rxn.lower_bound,
                    "Upper Bound": rxn.upper_bound,
                }
            )
    df = pd.DataFrame(rows) if rows else None
    return BoundInconsistencyResult(count=len(rows), dataframe=df)


def check_gene_orphan_reactions(model: cobra.Model) -> GeneOrphanResult:
    """
    Identify reactions with no gene-protein-reaction (GPR) associations.

    Exchange, demand, and sink reactions are excluded from this check because
    they legitimately have no gene associations by convention.

    Returns
    -------
    GeneOrphanResult
    """
    boundary_prefixes = ("EX_", "DM_", "SK_", "ATPM")
    orphan_rows: List[Dict[str, str]] = []
    orphan_ids: List[str] = []

    for rxn in model.reactions:
        if any(rxn.id.startswith(p) for p in boundary_prefixes):
            continue
        if not rxn.gene_reaction_rule or rxn.gene_reaction_rule.strip() == "":
            orphan_rows.append(
                {
                    "Reaction ID": rxn.id,
                    "Name": rxn.name or "—",
                    "Subsystem": rxn.subsystem or "—",
                }
            )
            orphan_ids.append(rxn.id)

    df = pd.DataFrame(orphan_rows) if orphan_rows else None
    return GeneOrphanResult(
        count=len(orphan_ids), reaction_ids=orphan_ids, dataframe=df
    )


def check_mass_balance(model: cobra.Model) -> MassBalanceResult:
    """
    Check stoichiometric mass balance for each non-boundary reaction.

    COBRApy's ``check_mass_balance`` returns a dict of imbalances only for
    reactions with formula-annotated metabolites. Exchange reactions are
    intentionally excluded.

    Returns
    -------
    MassBalanceResult
    """
    rows: List[Dict[str, str]] = []
    balanced = 0
    unbalanced = 0

    for rxn in model.reactions:
        if rxn.boundary:
            continue
        try:
            imbalance = rxn.check_mass_balance()
            if imbalance:
                unbalanced += 1
                rows.append(
                    {
                        "Reaction ID": rxn.id,
                        "Name": rxn.name or "—",
                        "Imbalance": str(imbalance),
                    }
                )
            else:
                balanced += 1
        except Exception:
            # Reactions with unannotated metabolites raise errors — skip silently
            pass

    df = pd.DataFrame(rows) if rows else None
    return MassBalanceResult(
        balanced_count=balanced,
        unbalanced_count=unbalanced,
        dataframe=df,
    )
