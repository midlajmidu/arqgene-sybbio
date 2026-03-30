"""
backend/services/model_service.py
-----------------------------------
Service layer for SBML upload, parsing, and model summary extraction.

Delegates SBML I/O to core/model_loader.py and wraps the result in
Pydantic response models ready for FastAPI serialisation.
"""

from __future__ import annotations

import logging
import os
import tempfile

import cobra

from backend.schemas.responses import (
    CompartmentInfo,
    ErrorDetail,
    ModelSummaryResponse,
)
from backend.services.model_registry import get_registry
from backend.utils.solve_utils import configure_solver_in_context
from core.model_loader import extract_model_summary, load_model_from_path
from utils.solver_utils import detect_available_solvers, set_solver

logger = logging.getLogger(__name__)


def upload_and_register_model(
    file_bytes: bytes,
    filename: str,
    solver: str = "glpk",
    feasibility_tol: float = 1e-7,
    optimality_tol: float = 1e-7,
) -> ModelSummaryResponse:
    """
    Persist uploaded SBML bytes to a temp file, parse with COBRApy,
    configure solver, register in the model registry, and return a summary.

    Note on solver configuration at upload time
    --------------------------------------------
    Unlike FBA/pFBA, solver configuration at upload happens OUTSIDE a
    `with model:` block intentionally — this sets the PERSISTENT default
    solver on the registry model.  All subsequent FBA/pFBA calls will
    configure the solver INSIDE their own `with model:` contexts and roll
    it back, so the default set here acts as a sane starting point.

    Parameters
    ----------
    file_bytes : bytes
    filename : str
    solver : str
    feasibility_tol / optimality_tol : float

    Returns
    -------
    ModelSummaryResponse
    """
    # ------------------------------------------------------------------
    # 1. Write to named temp file (libSBML requires a filesystem path)
    # ------------------------------------------------------------------
    suffix = _infer_suffix(filename)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, mode="wb"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
    except OSError as exc:
        return _error_response("TEMP_FILE_ERROR", str(exc))

    # ------------------------------------------------------------------
    # 2. Parse SBML
    # ------------------------------------------------------------------
    try:
        model: cobra.Model = load_model_from_path(tmp_path)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return _error_response("SBML_PARSE_ERROR", str(exc))
    except Exception as exc:
        logger.exception("Unexpected model load error")
        return _error_response("UNKNOWN_LOAD_ERROR", str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 3. Set persistent default solver on the freshly loaded model.
    #    (Not inside `with model:` — this is the intentional registry default.)
    # ------------------------------------------------------------------
    available = detect_available_solvers()
    chosen_solver = solver if solver in available else available[0]
    try:
        set_solver(model, chosen_solver)
        from utils.solver_utils import configure_tolerance
        configure_tolerance(model, feasibility_tol, optimality_tol)
    except Exception as exc:
        logger.warning("Default solver configuration warning: %s", exc)

    # ------------------------------------------------------------------
    # 4. Extract summary metadata and register model
    # ------------------------------------------------------------------
    summary = extract_model_summary(model)
    model_id = get_registry().register(model)

    compartments = [
        CompartmentInfo(
            compartment_id=k,
            description=model.compartments.get(k, k),
        )
        for k in model.compartments
    ]

    return ModelSummaryResponse(
        success=True,
        message=f"Model '{summary.model_id}' loaded successfully.",
        model_id=model_id,
        internal_id=summary.model_id,
        num_reactions=summary.num_reactions,
        num_metabolites=summary.num_metabolites,
        num_genes=summary.num_genes,
        num_compartments=len(summary.compartments),
        compartments=compartments,
        objective_reaction=summary.objective_reaction,
        objective_direction=summary.objective_direction,
        exchange_reactions=summary.exchange_reactions,
        demand_reactions=summary.demand_reactions,
        sink_reactions=summary.sink_reactions,
        solver_name=chosen_solver,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_suffix(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext if ext else ".xml"


def _error_response(code: str, detail: str) -> ModelSummaryResponse:
    logger.error("[%s] %s", code, detail)
    return ModelSummaryResponse(
        success=False,
        message=f"Model upload failed: {code}",
        error=ErrorDetail(code=code, message=detail),
    )
