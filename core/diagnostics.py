"""
core/diagnostics.py
-------------------
Baseline FBA and pFBA diagnostics for genome-scale metabolic models.

Scientific purpose:
    Flux Balance Analysis (FBA) solves the linear programme:

        maximise   c^T v
        subject to S v = 0          (steady-state mass balance)
                   lb ≤ v ≤ ub      (thermodynamic / capacity bounds)

    where S is the m × n stoichiometric matrix, v the flux vector,
    and c the objective coefficient vector.

    Parsimonious FBA (pFBA) additionally minimises the sum of absolute
    fluxes among all optimal FBA solutions, yielding the most
    parsimonious flux distribution consistent with the objective.

    Both analyses are performed here and the results are packaged into
    typed dataclasses for clean downstream rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cobra
import cobra.flux_analysis
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------


@dataclass
class FBAResult:
    """Container for a single FBA or pFBA run."""

    analysis_type: str = "FBA"
    status: str = "not_run"
    objective_value: float = 0.0
    growth_rate: float = 0.0
    fluxes: Optional[pd.Series] = None
    top_reactions: Optional[pd.DataFrame] = None
    exchange_fluxes: Optional[pd.DataFrame] = None
    solver_name: str = "unknown"
    error_message: Optional[str] = None

    @property
    def is_optimal(self) -> bool:
        return self.status == "optimal"


@dataclass
class DiagnosticsReport:
    """Aggregated FBA & pFBA results."""

    fba: FBAResult = field(default_factory=FBAResult)
    pfba: FBAResult = field(default_factory=lambda: FBAResult(analysis_type="pFBA"))
    objective_reaction_id: str = "N/A"
    lp_formulation: str = ""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def run_diagnostics(model: cobra.Model) -> DiagnosticsReport:
    """
    Execute FBA and pFBA on the model and return a structured report.

    Parameters
    ----------
    model : cobra.Model
        A loaded COBRApy model with solver already configured.

    Returns
    -------
    DiagnosticsReport
    """
    report = DiagnosticsReport()
    report.objective_reaction_id = _get_objective_id(model)
    report.lp_formulation = _build_lp_description(model)

    report.fba = _run_fba(model)
    report.pfba = _run_pfba(model)

    return report


def _run_fba(model: cobra.Model) -> FBAResult:
    """
    Run standard Flux Balance Analysis (maximise / minimise objective).

    Returns
    -------
    FBAResult
    """
    result = FBAResult(analysis_type="FBA")
    try:
        with model:
            solution = model.optimize()
            result.status = solution.status
            result.solver_name = _solver_name(model)

            if solution.status == "optimal":
                result.objective_value = float(solution.objective_value)
                result.growth_rate     = float(solution.objective_value)
                result.fluxes          = solution.fluxes
                result.top_reactions   = _top_flux_table(model, solution.fluxes)
                result.exchange_fluxes = _extract_exchange_fluxes(model, solution.fluxes)

                # FIX 2: Guard against biologically impossible negative objective.
                # Causes: wrong objective direction ("min" instead of "max"),
                # reverse-coded biomass reaction (ub ≤ 0), or sign error in
                # objective coefficients.  We still return the numeric value so
                # the caller can display it, but populate error_message so the
                # API response includes a visible warning.
                if result.objective_value < -1e-6:
                    warning = (
                        f"FBA objective value is negative ({result.objective_value:.6f}). "
                        "Verify that the biomass reaction is forward-oriented (ub > 0) "
                        "and that objective direction is 'max'. "
                        "A reverse-coded biomass reaction is the most common cause."
                    )
                    logger.warning("[FBA NEGATIVE OBJECTIVE] %s", warning)
                    # Non-fatal — report the value but flag a warning via error_message.
                    result.error_message = warning
            else:
                result.error_message = (
                    f"FBA returned non-optimal status: {solution.status}"
                )
    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        logger.exception("FBA failed: %s", exc)

    return result


def _run_pfba(model: cobra.Model) -> FBAResult:
    """
    Run Parsimonious FBA (pFBA).

    pFBA first maximises the objective, then minimises the L1-norm of
    fluxes to select the simplest flux distribution.  The second step
    is implemented as an additional LP/QP solve by COBRApy.

    Returns
    -------
    FBAResult
    """
    result = FBAResult(analysis_type="pFBA")
    try:
        with model:
            pfba_soln = cobra.flux_analysis.pfba(model)
            result.status = pfba_soln.status
            result.solver_name = _solver_name(model)

            if pfba_soln.status == "optimal":
                result.objective_value = float(pfba_soln.objective_value)
                result.growth_rate     = float(pfba_soln.objective_value)
                result.fluxes          = pfba_soln.fluxes
                result.top_reactions   = _top_flux_table(model, pfba_soln.fluxes)
                result.exchange_fluxes = _extract_exchange_fluxes(model, pfba_soln.fluxes)
            else:
                result.error_message = (
                    f"pFBA returned non-optimal status: {pfba_soln.status}"
                )
    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        logger.warning("pFBA failed (non-fatal): %s", exc)

    return result


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def _top_flux_table(
    model: cobra.Model, fluxes: pd.Series, n: int = 10
) -> pd.DataFrame:
    """
    Return a DataFrame of the top-N flux-carrying reactions by absolute flux.
    Now includes Equation and Subsystem as requested in Section 4.
    """
    active = fluxes[fluxes.abs() > 1e-6].copy()
    top = active.abs().nlargest(n)

    data = []
    for rid in top.index:
        rxn = model.reactions.get_by_id(rid)
        data.append({
            "Reaction ID": rid,
            "Equation": rxn.build_reaction_string(use_metabolite_names=False),
            "Flux (mmol/gDW/h)": round(float(fluxes[rid]), 4),
            "|Flux|": round(float(abs(fluxes[rid])), 4),
            "Subsystem": rxn.subsystem or "N/A"
        })

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values("|Flux|", ascending=False).reset_index(drop=True)
    return df


def _extract_exchange_fluxes(model: cobra.Model, fluxes: pd.Series) -> pd.DataFrame:
    """
    Extract structured metadata for all exchange reactions with non-zero flux.

    Scientific rules:
    - Use model.exchanges if available, fallback to rxn.boundary.
    - Tolerance derived from model.solver.configuration.tolerances.feasibility.
    - Metabolite names extracted directly from metabolite objects.
    - Direction determined by signs relative to standard convention.
    """
    # 1. Determine dynamic threshold from solver state
    try:
        threshold = float(model.solver.configuration.tolerances.feasibility)
    except (AttributeError, ValueError):
        threshold = 1e-7

    # 2. Identify exchange reactions
    if hasattr(model, "exchanges") and model.exchanges:
        ex_rxns = model.exchanges
    else:
        ex_rxns = [r for r in model.reactions if r.boundary]

    exchange_data = []
    for rxn in ex_rxns:
        val = float(fluxes.get(rxn.id, 0.0))

        if abs(val) > threshold:
            # Metabolite metadata
            met_objs = list(rxn.metabolites.keys())
            if met_objs:
                met = met_objs[0]
                met_name_display = met.name or met.id
            else:
                met_name_display = "Unknown Metabolite"

            direction = "Uptake" if val < 0 else "Secretion"

            exchange_data.append({
                "metabolite_name": met_name_display,
                "reaction_id": rxn.id,
                "flux": round(val, 4),
                "direction": direction,
                "abs_flux": abs(val)
            })

    df = pd.DataFrame(exchange_data)
    if not df.empty:
        df = df.sort_values("abs_flux", ascending=False).drop(columns=["abs_flux"]).reset_index(drop=True)

    return df


def _get_objective_id(model: cobra.Model) -> str:
    """Return a readable objective reaction identifier."""
    try:
        expr = model.objective.expression
        coeff_dict = expr.as_coefficients_dict()
        ids = [
            str(v).split("_forward")[0].split("_reverse")[0]
            for v, c in coeff_dict.items()
            if float(c) != 0
        ]
        return ", ".join(sorted(set(ids))) or "None"
    except Exception:
        return "Undetermined"


def _solver_name(model: cobra.Model) -> str:
    """Extract a clean solver name from the model's interface."""
    try:
        return model.solver.interface.__name__.split(".")[-2]
    except Exception:
        return "unknown"


def _build_lp_description(model: cobra.Model) -> str:
    """
    Build a human-readable mathematical description of the FBA LP.

    This text is displayed in the expandable 'LP Formulation' section
    of the UI to aid scientific transparency.
    """
    m = len(model.metabolites)
    n = len(model.reactions)
    obj_dir = model.objective.direction
    obj_id = _get_objective_id(model)

    return f"""## FBA Linear Programme

**Objective function:**
```
{obj_dir.upper()} z = c^T · v
```
where **c** is the objective coefficient vector (1 for `{obj_id}`, 0 elsewhere).

**Subject to:**

| Constraint | Equation | Dimension |
|---|---|---|
| Steady-state mass balance | S · v = 0 | {m} constraints |
| Flux bounds | lb ≤ v ≤ ub | {n} variables |

**Problem dimensions:**
- Metabolites (rows in S): **{m}**
- Reactions (columns in S): **{n}**
- Genes in model: **{len(model.genes)}**

**Solver interface:** `optlang` wrapping the selected LP backend.

The stoichiometric matrix **S** encodes the network topology.
Exchange reactions (EX_) provide the interface between the cell
and the environment via uptake/secretion bounds.
"""
