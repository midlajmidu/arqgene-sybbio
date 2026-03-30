"""
core/model_loader.py
--------------------
Handles SBML file I/O, model loading via COBRApy, and extraction of a
structured model summary for display in the Streamlit UI.

Scientific purpose:
    The Systems Biology Markup Language (SBML) is the de-facto standard for
    encoding genome-scale metabolic models (GEMs). This module provides a
    robust loading pipeline that sanitises the file path, delegates parsing to
    libSBML (via cobra.io.read_sbml_model), and packages the loaded model's
    metadata into a typed dict for downstream analysis.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import IO, Optional

import cobra
import cobra.io

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class ModelSummary:
    """
    Lightweight, serialisable summary of a cobra.Model.

    All attributes mirror properties inspectable from the model object
    without running any LP solver, ensuring this summary is always cheap
    to compute.
    """

    model_id: str = "N/A"
    num_reactions: int = 0
    num_metabolites: int = 0
    num_genes: int = 0
    objective_reaction: str = "N/A"
    objective_direction: str = "N/A"
    compartments: list = field(default_factory=list)
    exchange_reactions: int = 0
    demand_reactions: int = 0
    sink_reactions: int = 0

    def as_dict(self) -> dict:
        """Return the summary as a plain dictionary for display."""
        return {
            "Model ID": self.model_id,
            "Reactions": self.num_reactions,
            "Metabolites": self.num_metabolites,
            "Genes": self.num_genes,
            "Objective Reaction": self.objective_reaction,
            "Objective Direction": self.objective_direction,
            "Compartments": ", ".join(self.compartments) if self.compartments else "N/A",
            "Exchange Reactions": self.exchange_reactions,
            "Demand Reactions": self.demand_reactions,
            "Sink Reactions": self.sink_reactions,
        }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def load_model_from_upload(uploaded_file: IO) -> cobra.Model:
    """
    Load a COBRApy model from a Streamlit UploadedFile object.

    The file is saved to a named temporary file on disk because libSBML
    requires a real filesystem path rather than an in-memory buffer.

    Parameters
    ----------
    uploaded_file : IO
        Streamlit ``UploadedFile`` (or any file-like with ``.read()``).

    Returns
    -------
    cobra.Model
        The parsed and validated COBRApy model.

    Raises
    ------
    ValueError
        If the file content cannot be parsed as valid SBML.
    RuntimeError
        If libSBML raises an unexpected error during parsing.
    """
    # Persist upload to a temp file with the correct .xml extension so
    # libSBML MIME detection works correctly.
    suffix = _infer_suffix(uploaded_file.name)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, mode="wb"
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        model = _parse_sbml(tmp_path)
    finally:
        # Always clean up the temp file to avoid disk leaks
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return model


def load_model_from_path(file_path: str) -> cobra.Model:
    """
    Load a COBRApy model from an absolute filesystem path.

    Parameters
    ----------
    file_path : str
        Absolute path to an SBML (.xml) file.

    Returns
    -------
    cobra.Model
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    return _parse_sbml(file_path)


def extract_model_summary(model: cobra.Model) -> ModelSummary:
    """
    Extract a structured summary from a loaded cobra.Model.

    Parameters
    ----------
    model : cobra.Model

    Returns
    -------
    ModelSummary
    """
    # Determine the active objective reaction(s)
    obj_reaction = _get_objective_reaction(model)

    # FIX 1: Use structural boundary detection (rxn.boundary) instead of ID-prefix
    # matching. rxn.boundary == True for any reaction with a single metabolite
    # (open-boundary topology), which is COBRApy's authoritative exchange test.
    # This matches what medium_service.py uses and prevents under/over-counting
    # on models that don't follow strict BiGG EX_ / DM_ / SK_ naming conventions.
    exchange = [r for r in model.reactions if r.boundary
                and not r.id.startswith(("DM_", "SK_"))]
    demand   = [r for r in model.reactions if r.id.startswith("DM_")]
    sink     = [r for r in model.reactions if r.id.startswith("SK_")]

    # FIX 4 + FIX 7: run structural sanity checks and log any issues as warnings.
    # These do not mutate the model or the response schema but will surface in
    # server logs so operators can detect malformed uploads.
    _validate_bounds(model)
    _validate_compartments(model)

    return ModelSummary(
        model_id=model.id or "unnamed_model",
        num_reactions=len(model.reactions),
        num_metabolites=len(model.metabolites),
        num_genes=len(model.genes),
        objective_reaction=obj_reaction,
        objective_direction=model.objective.direction,
        compartments=list(model.compartments.keys()),
        exchange_reactions=len(exchange),
        demand_reactions=len(demand),
        sink_reactions=len(sink),
    )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _parse_sbml(path: str) -> cobra.Model:
    """Delegate SBML parsing to COBRApy with informative error wrapping."""
    try:
        model = cobra.io.read_sbml_model(path)
    except cobra.io.sbml.CobraSBMLError as exc:
        raise ValueError(
            f"SBML parsing error — the file may be malformed or use an "
            f"unsupported SBML level/version.\n\nDetails: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error while loading SBML model: {exc}"
        ) from exc

    if model is None:
        raise ValueError("COBRApy returned an empty model. Is the file valid SBML?")

    return model


def _get_objective_reaction(model: cobra.Model) -> str:
    """Return a human-readable representation of the current objective."""
    try:
        expr = model.objective.expression
        # For linear objectives, extract variable names with non-zero coefficients
        vars_with_coeff = {
            str(v): float(c)
            for v, c in expr.as_coefficients_dict().items()
            if float(c) != 0
        }
        if not vars_with_coeff:
            return "None"
        # Trim '_reverse_...'-suffixed duplicates that COBRApy adds internally
        clean = [k.split("_forward")[0].split("_reverse")[0] for k in vars_with_coeff]
        return ", ".join(sorted(set(clean)))
    except Exception:
        return "Undetermined"


def _infer_suffix(filename: Optional[str]) -> str:
    """Return file extension from filename or default to '.xml'."""
    if filename and "." in filename:
        return os.path.splitext(filename)[1]
    return ".xml"


# ---------------------------------------------------------------------------
# FIX 4 — Bound validity check
# ---------------------------------------------------------------------------


def _validate_bounds(model: cobra.Model) -> None:
    """
    Log warnings for reactions with invalid or extreme bounds.

    Checks (does NOT mutate the model):
    - lb > ub (infeasible by construction)
    - NaN / None bounds (cause silent GLPK failures)
    - |bound| > 1e6 (risk of unbounded LP with GLPK)

    Called from extract_model_summary() at upload time so issues appear in
    server logs before any LP solve is attempted.
    """
    import math
    n_infeasible = 0
    n_nan = 0
    n_large = 0
    infeasible_examples: list = []

    for rxn in model.reactions:
        lb, ub = rxn.lower_bound, rxn.upper_bound

        # None / NaN bounds
        if lb is None or ub is None:
            n_nan += 1
            logger.warning(
                "Bound validation: reaction '%s' has None bound (lb=%s, ub=%s).",
                rxn.id, lb, ub,
            )
            continue
        try:
            lb_f, ub_f = float(lb), float(ub)
        except (TypeError, ValueError):
            n_nan += 1
            logger.warning(
                "Bound validation: reaction '%s' has non-numeric bound.", rxn.id
            )
            continue

        if math.isnan(lb_f) or math.isnan(ub_f) or math.isinf(lb_f) or math.isinf(ub_f):
            n_nan += 1
            logger.warning(
                "Bound validation: reaction '%s' has NaN/Inf bound (lb=%s, ub=%s). "
                "GLPK may report unbounded LP.",
                rxn.id, lb_f, ub_f,
            )
            continue

        # Infeasible: lb > ub
        if lb_f > ub_f + 1e-9:
            n_infeasible += 1
            if len(infeasible_examples) < 5:
                infeasible_examples.append(f"{rxn.id}(lb={lb_f:.2f},ub={ub_f:.2f})")

        # Extreme bounds
        if abs(lb_f) > 1e6 or abs(ub_f) > 1e6:
            n_large += 1

    if n_infeasible:
        logger.warning(
            "Bound validation: %d reaction(s) have lb > ub (infeasible by construction): %s",
            n_infeasible,
            ", ".join(infeasible_examples),
        )
    if n_nan:
        logger.warning(
            "Bound validation: %d reaction(s) have NaN/Inf/None bounds.", n_nan
        )
    if n_large:
        logger.warning(
            "Bound validation: %d reaction(s) have |bound| > 1e6. "
            "GLPK may produce unbounded LP solutions on these reactions.",
            n_large,
        )
    if not (n_infeasible or n_nan or n_large):
        logger.debug("Bound validation passed: all %d reaction bounds are finite and lb ≤ ub.",
                     len(model.reactions))


# ---------------------------------------------------------------------------
# FIX 7 — Compartment-metabolite cross-validation
# ---------------------------------------------------------------------------


def _validate_compartments(model: cobra.Model) -> None:
    """
    Log warnings for metabolites that reference undeclared compartments.

    Standard GEMs encode compartment in the metabolite ID suffix (e.g. _c, _e, _p)
    AND in the SBML compartment attribute.  A mismatch indicates a malformed model
    that may produce unexpected stoichiometry or silently wrong FBA results.

    Does NOT mutate the model.
    """
    declared = set(model.compartments.keys())
    orphan: list = []

    for met in model.metabolites:
        comp = getattr(met, "compartment", None)
        if comp and comp not in declared:
            orphan.append(f"{met.id}(comp={comp!r})")

    if orphan:
        example_str = ", ".join(orphan[:5])
        logger.warning(
            "Compartment validation: %d metabolite(s) reference undeclared compartment(s). "
            "Declared: %s. Examples: %s",
            len(orphan),
            sorted(declared),
            example_str + (" ..." if len(orphan) > 5 else ""),
        )
    else:
        logger.debug(
            "Compartment validation passed: all %d metabolites in declared compartments.",
            len(model.metabolites),
        )
