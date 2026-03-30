"""
backend/services/model_registry.py
------------------------------------
Thread-safe, in-memory model registry.

Scientific purpose:
    Genome-scale metabolic models are large objects (often 10–200 MB of
    parsed graph data).  Parsing SBML is slow (~1–30 s depending on model
    size), so we keep loaded cobra.Model instances alive in memory for the
    duration of a user session, keyed by a stable UUID.

Architecture:
    - Singleton ModelRegistry class, instantiated once at module level.
    - Uses threading.RLock for safe concurrent access (FastAPI runs multiple
      async worker threads under uvicorn).
    - Supports basic TTL cleanup to prevent unbounded memory growth.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, Optional, Tuple

import cobra

logger = logging.getLogger(__name__)

# How long (seconds) an unused model stays in memory before eviction.
_DEFAULT_TTL_SECONDS: int = 3600  # 1 hour


class ModelRegistry:
    """
    Singleton registry mapping UUID → (cobra.Model, last_access_timestamp).

    Usage
    -----
    registry = get_registry()         # always returns the same instance
    model_id = registry.register(model)
    model    = registry.get(model_id)
    registry.remove(model_id)
    registry.evict_expired()          # called periodically
    """

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._store: Dict[str, Tuple[cobra.Model, float]] = {}
        # Lightweight metadata dict — stores user-chosen objective, etc.
        # Never holds cobra objects; safe to read without acquiring _lock.
        self._metadata: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        logger.info("ModelRegistry initialised (TTL=%ds)", ttl_seconds)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register(self, model: cobra.Model) -> str:
        """
        Store a cobra.Model and return a new UUID key.

        Parameters
        ----------
        model : cobra.Model

        Returns
        -------
        str
            A UUID4 string that identifies this model in all subsequent calls.
        """
        model_id = str(uuid.uuid4())
        with self._lock:
            self._store[model_id] = (model, time.monotonic())
        logger.info("Registered model '%s' as %s", model.id, model_id)
        return model_id

    def get(self, model_id: str) -> Optional[cobra.Model]:
        """
        Retrieve a model by its UUID, refreshing its TTL on access.

        Returns None if the model_id is unknown or has been evicted.
        """
        with self._lock:
            entry = self._store.get(model_id)
            if entry is None:
                logger.warning("Model %s not found in registry.", model_id)
                return None
            model, _ = entry
            # Refresh last-access time (LRU-style)
            self._store[model_id] = (model, time.monotonic())
            return model

    def remove(self, model_id: str) -> bool:
        """
        Explicitly delete a model from the registry.

        Returns True if the model existed and was removed.
        """
        with self._lock:
            if model_id in self._store:
                model, _ = self._store.pop(model_id)
                self._metadata.pop(model_id, None)   # clean up metadata
                model_name = getattr(model, "id", model_id)
                logger.info("Removed model %s ('%s') from registry.", model_id, model_name)
                return True
            return False

    def evict_expired(self) -> int:
        """
        Remove all models whose last-access time exceeds TTL.

        Returns the number of models evicted.
        """
        now = time.monotonic()
        evicted = 0
        with self._lock:
            expired = [
                mid
                for mid, (_, ts) in self._store.items()
                if now - ts > self._ttl
            ]
            for mid in expired:
                model, _ = self._store.pop(mid)
                self._metadata.pop(mid, None)        # clean up metadata
                logger.info("Evicted stale model %s ('%s').", mid, model.id)
                evicted += 1
        if evicted:
            logger.info("Evicted %d stale model(s) from registry.", evicted)
        return evicted

    def list_ids(self) -> list[str]:
        """Return all currently registered model UUIDs."""
        with self._lock:
            return list(self._store.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def set_model_metadata(self, model_id: str, **kwargs) -> None:
        """
        Store arbitrary key-value metadata for a registered model.

        Silently ignores unknown model_ids (model may have been evicted).
        Thread-safe: acquires the registry lock.
        """
        with self._lock:
            if model_id in self._store:
                if model_id not in self._metadata:
                    self._metadata[model_id] = {}
                self._metadata[model_id].update(kwargs)

    def get_model_metadata(self, model_id: str) -> Dict:
        """
        Return a copy of stored metadata for `model_id`.

        Returns an empty dict if no metadata has been set or model is gone.
        """
        with self._lock:
            return dict(self._metadata.get(model_id, {}))


# ------------------------------------------------------------------
# Module-level singleton — import and call get_registry() anywhere
# ------------------------------------------------------------------

_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """
    Return the application-wide ModelRegistry singleton.

    Thread-safe double-checked locking ensures the registry is
    created exactly once even under concurrent startup.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry
