"""
backend/services/reactions_service.py
---------------------------------------
Service layer for the paginated reaction listing endpoint.
"""

from __future__ import annotations

import logging
from math import ceil
from typing import List, Optional

import cobra

from backend.exceptions import ModelNotFoundError
from backend.schemas.responses import ErrorDetail, ReactionEntry, ReactionsListResponse
from backend.services.model_registry import get_registry

logger = logging.getLogger(__name__)


def list_reactions(
    model_id: str,
    page: int = 1,
    page_size: int = 25,
    search: Optional[str] = None,
    subsystem: Optional[str] = None,
) -> ReactionsListResponse:
    """
    Return a paginated, optionally filtered list of reactions.

    Raises
    ------
    ModelNotFoundError
        Propagates to the route layer → HTTP 404.
    """
    model: Optional[cobra.Model] = get_registry().get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id)

    # Build entry list — read-only access, no `with model:` needed
    entries: List[ReactionEntry] = [
        ReactionEntry(
            id=r.id,
            name=r.name or "—",
            subsystem=r.subsystem or "—",
            lower_bound=r.lower_bound,
            upper_bound=r.upper_bound,
            gpr=r.gene_reaction_rule or "—",
            formula=r.reaction,
            num_metabolites=len(r.metabolites),
            num_genes=len(r.genes),
            is_boundary=r.boundary,
        )
        for r in model.reactions
    ]

    # Subsystem filter (exact match)
    if subsystem and subsystem.strip() and subsystem.lower() != "all":
        entries = [e for e in entries if e.subsystem == subsystem]

    # Free-text search (case-insensitive, multi-field)
    if search and search.strip():
        q = search.strip().lower()
        entries = [
            e
            for e in entries
            if q in e.id.lower()
            or q in e.name.lower()
            or q in e.subsystem.lower()
            or q in e.gpr.lower()
        ]

    total = len(entries)
    page_size = min(max(1, page_size), 200)
    total_pages = max(1, ceil(total / page_size))
    page = max(1, min(page, total_pages))

    start = (page - 1) * page_size
    page_entries = entries[start : start + page_size]

    return ReactionsListResponse(
        success=True,
        message=f"Page {page} of {total_pages} ({total} reactions matched).",
        model_id=model_id,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        reactions=page_entries,
    )
