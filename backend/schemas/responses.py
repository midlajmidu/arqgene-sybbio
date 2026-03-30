"""
backend/schemas/responses.py
-----------------------------
Pydantic response models for all API endpoints.

All responses include a `success` flag and a `message` field so the
frontend can handle both happy-path and error cases uniformly without
relying on HTTP status codes alone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------
# Shared primitives
# -----------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error payload embedded in every error response."""

    code: str
    message: str
    detail: Optional[str] = None


class BaseResponse(BaseModel):
    """Root response model — all endpoint responses extend this."""

    success: bool
    message: str


# -----------------------------------------------------------------------
# Model upload / summary
# -----------------------------------------------------------------------


class CompartmentInfo(BaseModel):
    compartment_id: str
    description: str


class ModelSummaryResponse(BaseResponse):
    """
    Returned by POST /upload-model.

    Contains enough metadata for the frontend to display the Model Summary
    tab without any direct COBRApy access.
    """

    model_id: str = Field("", description="Registry UUID for subsequent calls")
    internal_id: str = Field("", description="The model's own SBML id field")
    num_reactions: int = 0
    num_metabolites: int = 0
    num_genes: int = 0
    num_compartments: int = 0
    compartments: List[CompartmentInfo] = []
    objective_reaction: str = ""
    objective_direction: str = ""
    exchange_reactions: int = 0
    demand_reactions: int = 0
    sink_reactions: int = 0
    solver_name: str = ""
    error: Optional[ErrorDetail] = None


# -----------------------------------------------------------------------
# FBA / pFBA results
# -----------------------------------------------------------------------


class FluxReaction(BaseModel):
    """A single reaction with its steady-state flux value."""

    reaction_id: str
    equation: str = ""
    flux: float
    abs_flux: float
    subsystem: str = ""


class ExchangeFlux(BaseModel):
    """Structured data for the Exchange Flux Summary panel."""

    metabolite_name: str = Field(..., description="Common name of the exchanged metabolite")
    reaction_id: str = Field(..., description="The boundary reaction ID")
    flux: float = Field(..., description="Steady-state flux value (mmol·gDW⁻¹·h⁻¹)")
    direction: str = Field(..., description="Uptake or Secretion")


class FBAResponse(BaseResponse):
    """Returned by POST /run-fba."""

    model_id: str = ""
    analysis_type: str = "FBA"
    solver_status: str = ""
    solver_name: str = ""
    objective_value: float = 0.0
    growth_rate: float = 0.0
    top_reactions: List[FluxReaction] = []
    exchange_fluxes: List[ExchangeFlux] = []
    error: Optional[ErrorDetail] = None


class PFBAResponse(BaseResponse):
    """Returned by POST /run-pfba."""

    model_id: str = ""
    analysis_type: str = "pFBA"
    solver_status: str = ""
    solver_name: str = ""
    objective_value: float = 0.0
    growth_rate: float = 0.0
    total_absolute_flux: float = 0.0
    top_reactions: List[FluxReaction] = []
    exchange_fluxes: List[ExchangeFlux] = []
    error: Optional[ErrorDetail] = None


# -----------------------------------------------------------------------
# Validation results
# -----------------------------------------------------------------------


class BlockedReaction(BaseModel):
    reaction_id: str
    min_flux: float
    max_flux: float


class InconsistentBound(BaseModel):
    reaction_id: str
    name: str
    lower_bound: float
    upper_bound: float


class GeneOrphanReaction(BaseModel):
    reaction_id: str
    name: str
    subsystem: str


class MassImbalance(BaseModel):
    reaction_id: str
    name: str
    imbalance: str


class ValidationResponse(BaseResponse):
    """Returned by POST /validate-model."""

    model_id: str = ""
    is_feasible: bool = True
    feasibility_message: str = ""

    # Blocked reactions
    blocked_count: int = 0
    blocked_reactions: List[BlockedReaction] = []

    # Inconsistent bounds
    inconsistent_bounds_count: int = 0
    inconsistent_bounds: List[InconsistentBound] = []

    # Gene orphans
    gene_orphan_count: int = 0
    gene_orphan_reactions: List[GeneOrphanReaction] = []

    # Mass balance
    balanced_count: int = 0
    unbalanced_count: int = 0
    mass_imbalances: List[MassImbalance] = []

    # Summary warnings and errors
    warnings: List[str] = []
    errors: List[str] = []

    error: Optional[ErrorDetail] = None


# -----------------------------------------------------------------------
# Reactions listing (paginated)
# -----------------------------------------------------------------------


class ReactionEntry(BaseModel):
    id: str
    name: str
    subsystem: str
    lower_bound: float
    upper_bound: float
    gpr: str
    formula: str
    num_metabolites: int
    num_genes: int
    is_boundary: bool


class ReactionsListResponse(BaseResponse):
    """Returned by GET /reactions/{model_id}."""

    model_id: str = ""
    total: int = 0
    page: int = 1
    page_size: int = 25
    total_pages: int = 1
    reactions: List[ReactionEntry] = []
    error: Optional[ErrorDetail] = None


# -----------------------------------------------------------------------
# FVA results
# -----------------------------------------------------------------------


class FVAReactionResult(BaseModel):
    """
    FVA result for a single reaction.

    Scientific note:
        minimum and maximum bound the feasible flux range for this reaction
        at `fraction_of_optimum` of the primary objective.
        is_blocked == True means the reaction carries zero flux under ALL
        feasible conditions (within the FVA tolerance 1e-9).
    """

    reaction_id: str
    minimum: float
    maximum: float
    range: float          # maximum − minimum — 0 for blocked, large for flexible
    is_blocked: bool      # abs(min) < 1e-9 AND abs(max) < 1e-9


class FVAResponse(BaseResponse):
    """Returned by POST /models/{model_id}/fva."""

    model_id: str = ""
    objective_value: float = 0.0
    solver_status: str = ""
    solver_name: str = ""
    fraction_of_optimum: float = 1.0
    total_reactions: int = 0         # total reactions in full model
    analyzed_reactions: int = 0      # reactions included in this FVA run
    blocked_count: int = 0
    results: List[FVAReactionResult] = []
    error: Optional[ErrorDetail] = None


# -----------------------------------------------------------------------
# Objective switch result
# -----------------------------------------------------------------------


class ObjectiveSwitchResponse(BaseResponse):
    """Returned by POST /models/{model_id}/objective."""

    model_id: str = ""
    objective_reaction: str = ""
    direction: str = "max"
    baseline_objective_value: float = 0.0
    solver_status: str = ""
    solver_name: str = ""
    error: Optional[ErrorDetail] = None


# -----------------------------------------------------------------------
# Production envelope result
# -----------------------------------------------------------------------


class ProductionEnvelopeResponse(BaseResponse):
    """
    Returned by POST /models/{model_id}/production-envelope.

    Scientific note:
        Each (growth_values[i], product_values[i]) pair represents the
        maximum product flux achievable when biomass flux is constrained
        to ≥ growth_values[i].  Together these points trace the
        growth–production Pareto front (trade-off envelope).
    """

    model_id: str = ""
    biomass_reaction: str = ""
    product_reaction: str = ""
    max_growth: float = 0.0
    max_product: float = 0.0
    steps: int = 0
    growth_values: List[float] = []
    product_values: List[float] = []
    solver_status: str = ""
    solver_name: str = ""
    error: Optional[ErrorDetail] = None


# -----------------------------------------------------------------------
# Growth audit — zero-growth diagnostic
# -----------------------------------------------------------------------


class ExchangeStatus(BaseModel):
    """Snapshot of a single exchange reaction relevant to nutrient availability."""

    reaction_id: str
    lower_bound: float
    upper_bound: float
    uptake_enabled: bool  # True when lower_bound < 0  (nutrient can enter cell)
    present_in_model: bool = True


class BiomassAuditInfo(BaseModel):
    """Structural properties of the current biomass (objective) reaction."""

    reaction_id: str
    lower_bound: float
    upper_bound: float
    num_metabolites: int
    is_exchange: bool     # boundary reactions should NOT be the biomass
    has_metabolites: bool


class ATPMInfo(BaseModel):
    """ATP maintenance reaction (ATPM) status."""

    found: bool
    reaction_id: str = ""
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    maintenance_required: bool = False  # True when lower_bound > 0


class GrowthAuditResponse(BaseResponse):
    """
    Returned by GET /models/{model_id}/growth-audit.

    Contains a complete structured diagnosis of why biomass growth
    may be zero despite an optimal solver status.
    """

    model_id: str = ""
    biomass_reaction: Optional[BiomassAuditInfo] = None
    growth_rate: float = 0.0
    solver_status: str = ""

    # Nutrient availability
    carbon_sources: Dict[str, ExchangeStatus] = {}
    nitrogen_sources: Dict[str, ExchangeStatus] = {}
    phosphate_sulfate: Dict[str, ExchangeStatus] = {}
    oxygen: Optional[ExchangeStatus] = None

    # Maintenance demand
    atpm_bounds: ATPMInfo = Field(default_factory=lambda: ATPMInfo(found=False))

    # Exchange summary
    exchange_uptake_count: int = 0   # how many exchanges allow uptake
    total_exchanges: int = 0

    # FVA-based structural check (None if FVA could not run)
    biomass_structurally_blocked: Optional[bool] = None

    # Human-readable diagnosis
    likely_cause: str = ""

    # Step-by-step audit log
    audit_log: List[str] = []


# -----------------------------------------------------------------------
# Medium configuration
# -----------------------------------------------------------------------


class ExchangeEntry(BaseModel):
    """
    Single exchange reaction with both original (SBML) and effective bounds.

    effective_* = original_* overridden by any active medium modification.
    is_modified = True when the medium layer has changed this reaction's bounds.
    """

    reaction_id: str
    name: str
    formula: str = ""
    original_lower_bound: float
    original_upper_bound: float
    effective_lower_bound: float
    effective_upper_bound: float
    uptake_enabled: bool      # effective_lower_bound < 0
    is_modified: bool


class MediumResponse(BaseResponse):
    """
    Returned by GET, POST, and reset endpoints under /models/{model_id}/medium.

    Provides a full view of all exchange reactions, the effective medium
    bounds applied during solves, and summary stats for quick assessment.
    """

    model_id: str = ""
    total_exchanges: int = 0
    uptake_enabled_count: int = 0
    has_carbon_uptake: bool = False
    has_oxygen_uptake: bool = False
    exchanges: List[ExchangeEntry] = []
    modified_exchange_count: int = 0
    active_preset: Optional[str] = None


# -----------------------------------------------------------------------
# OptKnock strain design
# -----------------------------------------------------------------------


class OptKnockResponse(BaseResponse):
    """
    Returned by POST /models/{model_id}/optknock.

    Contains the identified knockout set, predicted phenotype metrics (from a
    JOINT final LP validate — both growth and product from the same solve context),
    comparison against baseline, step-by-step greedy search log, and algorithm
    transparency fields for reproducibility.
    """

    model_id: str = ""

    # Knockout results
    knocked_reactions: List[str] = []      # [] if no improvement found
    predicted_growth: float = 0.0         # max growth after KOs (h⁻¹)  │
    predicted_product_flux: float = 0.0  # product at 0.999×predicted_growth│ ← JOINT LP

    # Baseline comparisons
    baseline_growth: float = 0.0
    baseline_product_flux: float = 0.0

    # Configuration echoed back
    growth_fraction: float = 0.1

    # Improvement metric
    fold_improvement: float = 1.0         # final_product / baseline_product

    # Solver info
    solver_status: str = ""

    # Search statistics
    candidates_tested: int = 0
    essential_excluded: int = 0

    # Greedy round-by-round log
    search_log: List[str] = []

    # FIX 10 — Reproducibility / transparency fields
    algorithm_type: str = "greedy_lp_heuristic"    # never changes; documents what was run
    growth_floor_used: float = 0.0                 # actual hard floor applied (h⁻¹)
    essential_reactions_filtered: int = 0          # alias of essential_excluded for clarity
    iterations: int = 0                            # greedy rounds actually executed

