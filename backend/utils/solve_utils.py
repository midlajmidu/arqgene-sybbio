from __future__ import annotations
import logging
from typing import Optional
import cobra
from backend.exceptions import SolverNotAllowedError

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SOLVER_TIMEOUT = 60.0
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
VALIDATION_SOLVER_TIMEOUT = 600.0

ALLOWED_SOLVERS = ["glpk", "cplex", "gurobi", "highs"]

def validate_solver(solver: str) -> None:
    """Validate that the requested solver is on the allowlist."""
    if solver.lower() not in ALLOWED_SOLVERS:
        raise SolverNotAllowedError(solver)

def configure_solver_in_context(
    model: cobra.Model,
    solver_name: str,
    feasibility_tol: Optional[float] = None,
    optimality_tol: Optional[float] = None,
) -> None:
    """Configure solver interface and tolerances on a model inside a 'with model:' context."""
    try:
        model.solver = solver_name.lower()
    except Exception as exc:
        logger.error("Failed to set solver %s in context: %s", solver_name, exc)
        raise exc

    try:
        if feasibility_tol is not None:
            model.solver.configuration.tolerances.feasibility = feasibility_tol
        if optimality_tol is not None:
            model.solver.configuration.tolerances.optimality = optimality_tol
    except Exception as exc:
        logger.warning("Failed to configure tolerances in context: %s", exc)
