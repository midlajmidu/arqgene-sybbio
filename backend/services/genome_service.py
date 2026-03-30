"""
backend/services/genome_service.py
------------------------------------
Service layer for genome-to-model reconstruction.

Manages asynchronous reconstruction jobs with status tracking.
Each job progresses through stages:
  pending → parsing → annotating → mapping → building → gap_filling → exporting → completed

Thread-safe job store allows the frontend to poll for progress.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.services.model_registry import get_registry
from backend.services.model_service import upload_and_register_model

logger = logging.getLogger(__name__)


# =====================================================================
# Job data structures
# =====================================================================

@dataclass
class GenomeJob:
    """Tracks a single genome reconstruction job."""
    job_id: str
    filename: str
    status: str = "pending"  # pending|parsing|annotating|mapping|building|gap_filling|exporting|completed|failed
    progress: float = 0.0   # 0.0 → 1.0
    message: str = "Job queued"
    
    # Results (populated on completion)
    model_id: Optional[str] = None      # registry UUID
    sbml_path: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    
    # Errors
    error: Optional[str] = None
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# =====================================================================
# Job Store (in-process, thread-safe)
# =====================================================================

class GenomeJobStore:
    """Thread-safe storage for genome reconstruction jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, GenomeJob] = {}
        self._lock = threading.RLock()

    def create_job(self, filename: str) -> GenomeJob:
        job = GenomeJob(
            job_id=str(uuid.uuid4()),
            filename=filename,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[GenomeJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    if hasattr(job, k):
                        setattr(job, k, v)

    def list_jobs(self) -> List[GenomeJob]:
        with self._lock:
            return list(self._jobs.values())


# Module-level singleton
_job_store: Optional[GenomeJobStore] = None
_store_lock = threading.Lock()


def get_job_store() -> GenomeJobStore:
    global _job_store
    if _job_store is None:
        with _store_lock:
            if _job_store is None:
                _job_store = GenomeJobStore()
    return _job_store


# =====================================================================
# Pipeline execution
# =====================================================================

# Default path for the DIAMOND database (UniProt/SwissProt enzymes)
DEFAULT_DIAMOND_DB = os.path.abspath("data/reference/reference_db.dmnd")


def start_genome_reconstruction(
    file_bytes: bytes,
    filename: str,
    solver: str = "glpk",
    diamond_db_path: Optional[str] = DEFAULT_DIAMOND_DB,
) -> str:
    """
    Start an async genome reconstruction job.

    Returns the job_id for status polling.
    """
    store = get_job_store()
    job = store.create_job(filename)

    # Run in a background thread
    thread = threading.Thread(
        target=_run_reconstruction,
        args=(job.job_id, file_bytes, filename, solver, diamond_db_path),
        daemon=True,
    )
    thread.start()

    logger.info("Started genome reconstruction job %s for file '%s'", job.job_id, filename)
    return job.job_id


def _run_reconstruction(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    solver: str,
    diamond_db_path: Optional[str],
) -> None:
    """Background thread that runs the full reconstruction pipeline."""
    from core.genome_pipeline import run_full_pipeline, ReconstructionReport

    store = get_job_store()

    def _progress(message: str, fraction: float):
        # Map fraction to status
        if fraction < 0.10:
            status = "parsing"
        elif fraction < 0.50:
            status = "annotating"
        elif fraction < 0.70:
            status = "mapping"
        elif fraction < 0.80:
            status = "building"
        elif fraction < 0.90:
            status = "gap_filling"
        elif fraction < 1.0:
            status = "exporting"
        else:
            status = "completed"

        store.update_job(
            job_id,
            status=status,
            progress=fraction,
            message=message,
        )

    try:
        fasta_content = file_bytes.decode("utf-8", errors="replace")

        sbml_path, report = run_full_pipeline(
            fasta_content=fasta_content,
            filename=filename,
            diamond_db_path=diamond_db_path,
            progress_callback=_progress,
        )

        # Register the model in the existing platform
        model_id = None
        if sbml_path and os.path.exists(sbml_path):
            with open(sbml_path, "rb") as f:
                sbml_bytes = f.read()

            # Use the existing upload pipeline to register
            from backend.services.model_service import upload_and_register_model
            response = upload_and_register_model(
                file_bytes=sbml_bytes,
                filename=os.path.basename(sbml_path),
                solver=solver,
            )

            if response.success:
                model_id = response.model_id
                logger.info("Registered reconstructed model as %s", model_id)
            else:
                logger.warning("Model registration failed: %s", response.message)
                report.warnings.append(f"Model registration issue: {response.message}")

        # Convert report to dict
        report_dict = {
            "organism_name": report.organism_name,
            "input_type": report.input_type,
            "total_sequences": report.total_sequences,
            "avg_sequence_length": round(report.avg_sequence_length, 1),
            "annotated_genes": report.annotated_genes,
            "unique_ec_numbers": report.unique_ec_numbers,
            "annotation_coverage": round(report.annotation_coverage, 1),
            "mapped_reactions": report.mapped_reactions,
            "total_metabolites": report.total_metabolites,
            "gpr_associations": report.gpr_associations,
            "num_reactions": report.num_reactions,
            "num_metabolites": report.num_metabolites,
            "num_genes": report.num_genes,
            "num_compartments": report.num_compartments,
            "num_exchange_reactions": report.num_exchange_reactions,
            "gap_filled_reactions": report.gap_filled_reactions,
            "biomass_feasible": report.biomass_feasible,
            "growth_rate": round(report.growth_rate, 6),
            "total_time_seconds": round(report.total_time_seconds, 1),
            "warnings": report.warnings,
            "errors": report.errors,
        }

        store.update_job(
            job_id,
            status="completed",
            progress=1.0,
            message=f"Reconstruction complete: {report.num_reactions} reactions, growth={report.growth_rate:.4f}",
            model_id=model_id,
            sbml_path=sbml_path,
            report=report_dict,
            completed_at=time.time(),
        )

    except Exception as e:
        logger.exception("Genome reconstruction failed for job %s", job_id)
        store.update_job(
            job_id,
            status="failed",
            progress=0.0,
            message=f"Pipeline failed: {str(e)}",
            error=str(e),
            completed_at=time.time(),
        )
