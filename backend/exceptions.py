"""
backend/exceptions.py
----------------------
Domain-specific exception hierarchy for the SynB backend.

Using typed exceptions instead of sentinel return values lets the
route layer convert errors to proper HTTP status codes without any
coupling between the service layer and FastAPI.

Service functions raise these; route handlers catch and convert them.
"""

from __future__ import annotations


class SynBError(Exception):
    """Base class for all SynB backend errors."""


class ModelNotFoundError(SynBError):
    """
    Raised when a model UUID is not present in the ModelRegistry.

    HTTP mapping: 404 Not Found
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found in registry.")


class SolverNotAllowedError(SynBError):
    """
    Raised when a solver name is not on the server-side allowlist.

    HTTP mapping: 422 Unprocessable Entity
    """

    def __init__(self, solver: str) -> None:
        self.solver = solver
        super().__init__(
            f"Solver '{solver}' is not allowed. "
            f"Permitted values: glpk, cplex, gurobi."
        )


class InfeasibleModelError(SynBError):
    """
    Raised when a model LP is proven infeasible during analysis.

    HTTP mapping: 422 Unprocessable Entity
    """

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"Model LP is infeasible. {detail}".strip())


class SolverTimeoutError(SynBError):
    """
    Raised (conceptually) when a solver exceeds the configured wall-clock limit.

    In practice the asyncio.TimeoutError bubbles up from wait_for() in the
    route layer; this class exists for documentation and future use.

    HTTP mapping: 408 Request Timeout
    """


class ModelTooLargeError(SynBError):
    """
    Raised when a full-model FVA is requested on a large model (> 2 000 reactions)
    without the user explicitly acknowledging the runtime cost.

    HTTP mapping: 400 Bad Request (user must set confirm_full_model=True)
    """

    def __init__(self, n_reactions: int, limit: int = 2_000) -> None:
        self.n_reactions = n_reactions
        self.limit = limit
        super().__init__(
            f"Model has {n_reactions} reactions. "
            f"Full FVA may take several minutes. "
            f"Set confirm_full_model=True to proceed."
        )


class ReactionNotFoundError(SynBError):
    """
    Raised when a reaction_id does not exist in the model.

    HTTP mapping: 404 Not Found
    """

    def __init__(self, reaction_id: str) -> None:
        self.reaction_id = reaction_id
        super().__init__(f"Reaction '{reaction_id}' not found in this model.")


class ComputationTooExpensiveError(SynBError):
    """
    Raised when a computation (e.g. production envelope) is requested with
    parameters that would make it prohibitively slow on this node.

    HTTP mapping: 400 Bad Request — caller must reduce steps or model size.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
