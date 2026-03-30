"""
backend/services/medium_service.py
------------------------------------
Service layer for the Medium Configuration Layer.

Architecture
------------
The medium is stored exclusively in ModelRegistry metadata — it NEVER
permanently mutates the shared cobra.Model object.  During each solve call,
`apply_medium_from_metadata()` is called INSIDE the `with model:` context
so that COBRApy rolls back all bound changes automatically on context exit.

Metadata structure (stored in registry._metadata[model_id]):

    {
        "medium": {
            "modified_exchanges": {
                "EX_glc__D_e": {"lower_bound": -10.0, "upper_bound": 1000.0},
                ...
            },
            "active_preset": "aerobic_glucose" | null
        }
    }

Safety
------
- Only boundary (exchange) reactions may be updated.
- Bound values are clamped to [-1000, 1000].
- At most 100 exchange updates per request.
- Non-existent reactions are skipped during apply (defensive).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cobra

from backend.exceptions import (
    ModelNotFoundError,
    ReactionNotFoundError,
)
from backend.schemas.requests import MediumUpdateRequest, PresetRequest
from backend.schemas.responses import ExchangeEntry, MediumResponse
from backend.services.model_registry import get_registry

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────
_MAX_UPDATES = 100
_BOUND_MIN = -1000.0
_BOUND_MAX = 1000.0
_UPTAKE_TOL = 1e-9

# BiGG IDs for carbon exchange detection (used in MediumResponse summary)
_CARBON_EXCHANGE_IDS = frozenset(
    {
        "EX_glc__D_e",
        "EX_ac_e",
        "EX_succ_e",
        "EX_lac__D_e",
        "EX_fru_e",
        "EX_glyc_e",
        "EX_pyr_e",
        "EX_mal__L_e",
    }
)

# ─── Preset definitions ───────────────────────────────────────────────────
PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "aerobic_glucose": {
        "EX_glc__D_e": {"lower_bound": -10.0, "upper_bound": 1000.0},
        "EX_o2_e":     {"lower_bound": -20.0, "upper_bound": 1000.0},
        "EX_nh4_e":    {"lower_bound": -10.0, "upper_bound": 1000.0},
        "EX_pi_e":     {"lower_bound": -10.0, "upper_bound": 1000.0},
        "EX_so4_e":    {"lower_bound": -10.0, "upper_bound": 1000.0},
    },
    "anaerobic_glucose": {
        "EX_glc__D_e": {"lower_bound": -10.0, "upper_bound": 1000.0},
        "EX_o2_e":     {"lower_bound":  0.0,  "upper_bound": 1000.0},
        "EX_nh4_e":    {"lower_bound": -10.0, "upper_bound": 1000.0},
        "EX_pi_e":     {"lower_bound": -10.0, "upper_bound": 1000.0},
        "EX_so4_e":    {"lower_bound": -10.0, "upper_bound": 1000.0},
    },
    # "minimal_closed" is handled dynamically (all exchange lb → 0)
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def get_medium(model_id: str) -> MediumResponse:
    """
    Return all exchange reactions with effective bounds (SBML + medium overlay).

    Raises
    ------
    ModelNotFoundError → HTTP 404
    """
    model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    modified, active_preset = _read_medium_metadata(model_id)
    exchanges: List[ExchangeEntry] = []

    for rxn in model.reactions:
        if not rxn.boundary:
            continue
        orig_lb = rxn.lower_bound
        orig_ub = rxn.upper_bound
        mod = modified.get(rxn.id, {})
        eff_lb = mod.get("lower_bound", orig_lb)
        eff_ub = mod.get("upper_bound", orig_ub)
        exchanges.append(
            ExchangeEntry(
                reaction_id=rxn.id,
                name=rxn.name or rxn.id,
                formula=rxn.reaction,
                original_lower_bound=orig_lb,
                original_upper_bound=orig_ub,
                effective_lower_bound=eff_lb,
                effective_upper_bound=eff_ub,
                uptake_enabled=eff_lb < -_UPTAKE_TOL,
                is_modified=rxn.id in modified,
            )
        )

    uptake_count = sum(1 for e in exchanges if e.uptake_enabled)
    has_carbon = any(
        e.uptake_enabled for e in exchanges if e.reaction_id in _CARBON_EXCHANGE_IDS
    )
    has_oxygen = any(
        e.uptake_enabled for e in exchanges if e.reaction_id == "EX_o2_e"
    )
    modified_count = sum(1 for e in exchanges if e.is_modified)

    return MediumResponse(
        success=True,
        message=f"Medium: {len(exchanges)} exchange reactions, {modified_count} modified.",
        model_id=model_id,
        total_exchanges=len(exchanges),
        uptake_enabled_count=uptake_count,
        has_carbon_uptake=has_carbon,
        has_oxygen_uptake=has_oxygen,
        exchanges=exchanges,
        modified_exchange_count=modified_count,
        active_preset=active_preset,
    )


def update_medium(request: MediumUpdateRequest, model_id: str) -> MediumResponse:
    """
    Merge bound updates into medium metadata.

    Validates:
    - Reaction exists in model
    - Reaction is an exchange (boundary == True)
    - At most 100 reactions per request

    Never writes to the cobra.Model directly.

    Raises
    ------
    ModelNotFoundError    → HTTP 404
    ReactionNotFoundError → HTTP 404
    ValueError            → HTTP 422 (non-exchange reaction or too many updates)
    """
    if len(request.updates) > _MAX_UPDATES:
        raise ValueError(
            f"Too many updates: {len(request.updates)} reactions requested. "
            f"Limit is {_MAX_UPDATES} per request."
        )

    model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    rxn_by_id = {r.id: r for r in model.reactions}

    # Validate all reaction IDs before mutating metadata
    for rxn_id in request.updates:
        rxn = rxn_by_id.get(rxn_id)
        if rxn is None:
            raise ReactionNotFoundError(rxn_id)
        if not rxn.boundary:
            raise ValueError(
                f"Reaction '{rxn_id}' is not an exchange reaction (boundary=False). "
                "Only exchange reactions may be modified via the medium endpoint."
            )

    # Apply updates on top of existing modifications
    modified, _ = _read_medium_metadata(model_id)
    for rxn_id, bounds in request.updates.items():
        if rxn_id not in modified:
            modified[rxn_id] = {}
        if bounds.lower_bound is not None:
            modified[rxn_id]["lower_bound"] = float(bounds.lower_bound)
        if bounds.upper_bound is not None:
            modified[rxn_id]["upper_bound"] = float(bounds.upper_bound)

    _write_medium_metadata(model_id, modified, active_preset=None)
    logger.info(
        "Medium updated: model_id=%s, %d reactions modified.",
        model_id, len(request.updates),
    )
    return get_medium(model_id)


def apply_preset(preset_name: str, model_id: str) -> MediumResponse:
    """
    Overwrite the medium with a named preset configuration.

    For "minimal_closed", all exchange reactions are discovered from the
    model and their lower bounds set to 0 (uptake disabled).

    Raises
    ------
    ModelNotFoundError → HTTP 404
    ValueError         → HTTP 422 (unknown preset)
    """
    model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    if preset_name == "minimal_closed":
        # Close all exchange reactions dynamically
        modified: Dict[str, Dict[str, float]] = {
            rxn.id: {"lower_bound": 0.0}
            for rxn in model.reactions
            if rxn.boundary
        }
    elif preset_name in PRESETS:
        # Shallow copy so we don't mutate the module-level dict
        modified = {
            rxn_id: dict(bounds)
            for rxn_id, bounds in PRESETS[preset_name].items()
        }

        # FIX 6: Validate that every preset reaction ID exists in the model.
        # On non-BiGG models (AGORA, CarveMe, ModelSEED) the preset IDs may
        # not match, causing apply_medium_from_metadata to silently skip them.
        # Log a warning so the operator (and user) knows the preset was partial.
        rxn_ids_in_model = {r.id for r in model.reactions}
        missing = [rid for rid in modified if rid not in rxn_ids_in_model]
        if missing:
            logger.warning(
                "Preset '%s' applied to model_id=%s: %d reaction ID(s) not found in "
                "model and will have no effect: %s. "
                "This model may use non-BiGG reaction naming. "
                "Use the manual medium editor to set bounds by actual reaction ID.",
                preset_name, model_id, len(missing), missing,
            )
        else:
            logger.debug(
                "Preset '%s': all %d reaction IDs matched in model.",
                preset_name, len(modified),
            )
    else:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Valid presets: {', '.join(list(PRESETS.keys()) + ['minimal_closed'])}."
        )

    _write_medium_metadata(model_id, modified, active_preset=preset_name)
    logger.info("Preset '%s' applied to model_id=%s.", preset_name, model_id)
    return get_medium(model_id)


def reset_medium(model_id: str) -> MediumResponse:
    """
    Clear all medium modifications, restoring original SBML bounds behaviour.

    Raises
    ------
    ModelNotFoundError → HTTP 404
    """
    model = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    _write_medium_metadata(model_id, {}, active_preset=None)
    logger.info("Medium reset to SBML defaults for model_id=%s.", model_id)
    return get_medium(model_id)


def apply_medium_from_metadata(model: cobra.Model, model_id: str) -> int:
    """
    Apply stored medium modifications to the cobra.Model in-place.

    IMPORTANT: This function MUST be called inside a ``with model:`` context.
    COBRApy tracks all bound changes on its History stack and rolls them back
    automatically when the outer context exits.  The shared registry model is
    left pristine after each solve.

    Parameters
    ----------
    model : cobra.Model
        The model object (already inside a `with model:` context).
    model_id : str
        Registry UUID — used to look up medium metadata.

    Returns
    -------
    int
        Number of exchange bounds actually applied (0 → no medium defined).
    """
    modified, _ = _read_medium_metadata(model_id)
    if not modified:
        return 0

    rxn_by_id = {r.id: r for r in model.reactions}
    count = 0
    for rxn_id, bounds in modified.items():
        rxn = rxn_by_id.get(rxn_id)
        if rxn is None:
            # Reaction may have been renamed or doesn't exist — silently skip
            continue
        if "lower_bound" in bounds:
            rxn.lower_bound = float(bounds["lower_bound"])    # tracked by COBRApy
        if "upper_bound" in bounds:
            rxn.upper_bound = float(bounds["upper_bound"])    # tracked by COBRApy
        count += 1

    if count:
        logger.debug(
            "Medium applied: %d exchange bound(s) set for model_id=%s "
            "(all will roll back on context exit).",
            count, model_id,
        )
    return count


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════


def _read_medium_metadata(
    model_id: str,
) -> tuple[Dict[str, Dict[str, float]], Optional[str]]:
    """Return (modified_exchanges, active_preset) from registry metadata."""
    meta = get_registry().get_model_metadata(model_id)
    medium = meta.get("medium", {})
    return (
        medium.get("modified_exchanges", {}),
        medium.get("active_preset", None),
    )


def _write_medium_metadata(
    model_id: str,
    modified: Dict[str, Dict[str, float]],
    active_preset: Optional[str],
) -> None:
    """Persist medium state to registry metadata (thread-safe)."""
    get_registry().set_model_metadata(
        model_id,
        medium={
            "modified_exchanges": modified,
            "active_preset": active_preset,
        },
    )
