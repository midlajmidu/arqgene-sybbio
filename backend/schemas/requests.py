"""
backend/schemas/requests.py
----------------------------
Pydantic request models for all API endpoints.

These models define what the frontend sends to the backend
and are validated automatically by FastAPI before delivery
to the service layer.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class FBARequest(BaseModel):
    """
    Input schema for POST /run-fba and POST /run-pfba.

    Attributes
    ----------
    model_id : str
        UUID returned by the /upload-model endpoint.
        Used to look up the cobra.Model in the in-memory registry.
    solver : str
        LP solver backend name. Defaults to 'glpk' (always available).
    feasibility_tol : float
        Primal feasibility tolerance for LP constraints.
    optimality_tol : float
        Dual optimality tolerance.
    """

    model_id: str = Field(..., description="UUID of a previously uploaded model")
    solver: str = Field("glpk", description="LP solver backend (glpk, cplex, gurobi)")
    feasibility_tol: float = Field(1e-7, ge=1e-12, le=1e-3, description="Feasibility tolerance")
    optimality_tol: float = Field(1e-7, ge=1e-12, le=1e-3, description="Optimality tolerance")


class ValidationRequest(BaseModel):
    """
    Input schema for POST /validate-model.

    Attributes
    ----------
    model_id : str
        UUID of the model to validate.
    run_fva : bool
        Whether to run Flux Variability Analysis for blocked-reaction detection.
        Disabled by default since FVA is expensive on large models.
    """

    model_id: str = Field(..., description="UUID of a previously uploaded model")
    run_fva: bool = Field(
        True,
        description="Run FVA for blocked reaction detection (slower but complete)",
    )


class FVARequest(BaseModel):
    """
    Input schema for POST /models/{model_id}/fva.

    Attributes
    ----------
    solver : str
        LP solver backend. Defaults to 'glpk'.
    fraction_of_optimum : float
        FVA is run at this fraction of the optimal objective value.
        1.0 = only flux distributions achieving the full optimum.
        0.0 = no optimality constraint (find all thermodynamically feasible ranges).
    reaction_ids : Optional[List[str]]
        Explicit subset of reaction IDs to analyse. If None, the full model is used.
    confirm_full_model : bool
        Must be True to proceed with full-model FVA on a model with > 2 000 reactions.
        Prevents accidentally triggering a 30-minute analysis.
    timeout : Optional[int]
        Per-request timeout override in seconds (capped at server maximum).
    """

    solver: str = Field("glpk", description="LP solver backend")
    fraction_of_optimum: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Fraction of optimum for FVA (0.0–1.0)",
    )
    reaction_ids: Optional[List[str]] = Field(
        None, description="Specific reaction IDs to analyse (None = full model)"
    )
    confirm_full_model: bool = Field(
        False,
        description="Set True to run full-model FVA on large models (> 2 000 reactions)",
    )
    timeout: Optional[int] = Field(
        None, ge=30, le=3600,
        description="Per-request timeout in seconds (server cap applies)",
    )


class ObjectiveSwitchRequest(BaseModel):
    """
    Input schema for POST /models/{model_id}/objective.

    Attributes
    ----------
    reaction_id : str
        ID of the reaction to set as the new objective.
    direction : Literal["max", "min"]
        Optimisation direction. Defaults to 'max' (maximise production).
    """

    reaction_id: str = Field(..., description="Reaction ID to use as objective")
    direction: Literal["max", "min"] = Field(
        "max", description="Optimisation direction: max or min"
    )


class ProductionEnvelopeRequest(BaseModel):
    """
    Input schema for POST /models/{model_id}/production-envelope.

    Attributes
    ----------
    product_reaction : str
        Reaction ID for the product of interest.
    biomass_reaction : str
        Reaction ID for biomass (growth objective).
    steps : int
        Number of growth rate points to sample (default 20, max 50).
    solver : str
        LP solver backend.
    """

    product_reaction: str = Field(..., description="Product reaction ID")
    biomass_reaction: str = Field(..., description="Biomass/growth reaction ID")
    steps: int = Field(20, ge=2, le=50, description="Number of growth-rate points to sample")
    solver: str = Field("glpk", description="LP solver backend")


# -----------------------------------------------------------------------
# Medium configuration
# -----------------------------------------------------------------------


class MediumBoundUpdate(BaseModel):
    """
    Flux bounds for a single exchange reaction in the medium.

    Both fields are optional — omitted fields are not changed.
    Values are clamped to [-1000, 1000] (COBRA standard range).
    """

    lower_bound: Optional[float] = Field(
        None, ge=-1000.0, le=1000.0,
        description="New lower bound; negative = uptake enabled",
    )
    upper_bound: Optional[float] = Field(
        None, ge=-1000.0, le=1000.0,
        description="New upper bound",
    )


class MediumUpdateRequest(BaseModel):
    """
    Input schema for POST /models/{model_id}/medium.

    Maps reaction_id → bound updates.  Limited to 100 exchanges per
    request to prevent oversized payloads.
    """

    updates: Dict[str, MediumBoundUpdate] = Field(
        ...,
        description="Map of {reaction_id: bound_update}. Max 100 entries.",
    )


class PresetRequest(BaseModel):
    """
    Input schema for POST /models/{model_id}/medium/preset.

    Presets:
        aerobic_glucose   — standard aerobic minimal medium with glucose
        anaerobic_glucose — same but O2 closed (anaerobic fermentation)
        minimal_closed    — all exchanges closed (lb = 0)
    """

    preset: Literal["aerobic_glucose", "anaerobic_glucose", "minimal_closed"] = Field(
        ..., description="Preset name to apply"
    )


# -----------------------------------------------------------------------
# OptKnock strain design
# -----------------------------------------------------------------------


class OptKnockRequest(BaseModel):
    """
    Input schema for POST /models/{model_id}/optknock.

    Greedy OptKnock-style sequential knockout search: finds up to
    `max_knockouts` reaction deletions that maximise product flux while
    maintaining biomass ≥ growth_fraction × baseline_growth.
    """

    biomass_reaction: str = Field(
        ..., description="ID of the biomass/growth reaction (must be internal, not exchange)"
    )
    product_reaction: str = Field(
        ..., description="ID of the target product reaction to maximise"
    )
    max_knockouts: int = Field(
        1, ge=1, le=3,
        description="Maximum number of reaction knockouts to test (1–3)",
    )
    solver: Optional[str] = Field(None, description="LP solver backend (default: glpk)")
    growth_fraction: float = Field(
        0.1, ge=0.01, le=1.0,
        description=(
            "Minimum growth fraction α: enforces biomass ≥ α × baseline_growth during search. "
            "Schema minimum is 0.01. The service applies an absolute hard floor of 1e-4 h⁻¹ "
            "regardless, preventing zero-growth product solutions."
        ),
    )
    candidate_reactions: Optional[List[str]] = Field(
        None,
        description=(
            "Optional explicit knockout candidate reaction IDs. "
            "If omitted, auto-selected (non-exchange, non-essential, ≤300)."
        ),
    )
    timeout: Optional[int] = Field(
        None, ge=30, le=600,
        description="Per-request timeout in seconds (max 600)",
    )
