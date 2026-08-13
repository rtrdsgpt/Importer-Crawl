"""
FastAPI wrapper around the pipeline in main.py's run_pipeline() --
Discovery -> Scraping -> [Directory & Report Mining] -> Ranking ->
[Validation]. Same phases, same incremental checkpointing to data/, same
provider set as the CLI (main.py) and Streamlit app (app.py); this just
adds an HTTP surface so the pipeline can be triggered and polled by another
service instead of a human at a terminal or browser.

A run can take minutes to hours (LLM calls + politely-delayed scraping), so
POST /discover returns a job id immediately and the pipeline runs in a
background thread; poll GET /jobs/{job_id} for status and
GET /jobs/{job_id}/results once it's done.

Usage:
    uvicorn api:app --reload
    curl -X POST localhost:8000/discover \\
        -H 'content-type: application/json' \\
        -d '{"product": "Ceramic Tiles", "country": "Germany", "provider": "groq"}'
    curl localhost:8000/jobs/<job_id>
    curl localhost:8000/jobs/<job_id>/results
"""

from __future__ import annotations

import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import rank_engine as rnk
from main import run_pipeline

app = FastAPI(
    title="Exporter Crawl API",
    description="Discover and rank importer companies for an Indian exporter, given a product and target country.",
    version="1.0.0",
)

# Bounds how many discovery runs execute concurrently -- each one makes
# many sequential, rate-limited network/LLM calls, so unbounded concurrency
# would mostly just contend for the same provider's quota rather than
# finish any faster.
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_CONCURRENT_JOBS", "2")))

JobStatus = Literal["pending", "running", "completed", "failed"]

# Progress lines kept per job, capped so a very long run can't grow this
# without bound in memory.
_MAX_PROGRESS_LINES = 500


@dataclass
class Job:
    id: str
    product: str
    country: str
    provider: str
    status: JobStatus = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    progress: list[str] = field(default_factory=list)
    results: Optional[list[dict]] = None
    error: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, msg: str) -> None:
        with self._lock:
            self.progress.append(msg)
            if len(self.progress) > _MAX_PROGRESS_LINES:
                self.progress = self.progress[-_MAX_PROGRESS_LINES:]


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


class DiscoverRequest(BaseModel):
    # populate_by_name lets callers send either "validate" (the wire name,
    # matching main.py's --validate CLI flag) or "validate_results" (the
    # Python attribute name below) -- BaseModel already defines a
    # deprecated validate() classmethod, so a field literally named
    # "validate" shadows it and triggers a pydantic UserWarning.
    model_config = ConfigDict(populate_by_name=True)

    product: str = Field(..., examples=["Ceramic Tiles"])
    country: str = Field(..., examples=["Germany"])
    provider: str = Field(
        "groq", description=f"One of {list(rnk.PROVIDER_CONFIGS)}. Default: groq (free)."
    )
    model: Optional[str] = Field(None, description="Defaults to the provider's default model.")
    api_key: Optional[str] = Field(
        None,
        description="Defaults to the provider's env var. Accepts a comma-separated list of "
                    "keys for round-robin rotation when one key's daily quota runs out mid-run.",
    )
    top_n: int = 10
    min_score: int = Field(40, ge=0, le=100)
    max_per_query: int = Field(8, ge=1, le=100)
    delay: float = Field(1.0, ge=0.0, description="Delay in seconds between outbound requests.")
    localize: bool = True
    mine_directories: bool = False
    validate_results: bool = Field(True, alias="validate")
    use_map_lookup: bool = True
    min_candidates: int = Field(15, ge=0)
    max_expansion_rounds: int = Field(2, ge=0)


class DiscoverResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    product: str
    country: str
    provider: str
    status: JobStatus
    created_at: str
    finished_at: Optional[str]
    progress_tail: list[str]
    error: Optional[str]


def _get_job(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job with id {job_id!r}")
    return job


def _resolve_api_key(provider: str, explicit: Optional[str]) -> str:
    config = rnk.PROVIDER_CONFIGS[provider]
    api_key = explicit or os.environ.get(config["env_var"]) or config.get("placeholder_key")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"No API key found for provider {provider!r}. Set {config['env_var']} in "
                   f"the server environment or pass api_key in the request body.",
        )
    return api_key


def _run_job(job: Job, req: DiscoverRequest, api_key: str) -> None:
    job.status = "running"
    try:
        results = run_pipeline(
            req.product, req.country, req.provider, req.model, api_key,
            max_per_query=req.max_per_query, delay=req.delay,
            top_n=req.top_n, min_score=req.min_score,
            localize=req.localize, mine_dirs=req.mine_directories,
            validate=req.validate_results, use_map_lookup=req.use_map_lookup,
            min_candidates=req.min_candidates, max_expansion_rounds=req.max_expansion_rounds,
            on_progress=job.log,
        )
        job.results = results
        job.status = "completed"
    except Exception as exc:  # noqa: BLE001 - surface any failure via job status, not a crashed thread
        job.error = str(exc)
        job.status = "failed"
    finally:
        job.finished_at = datetime.now(timezone.utc).isoformat()


@app.post("/discover", response_model=DiscoverResponse)
def discover(req: DiscoverRequest) -> DiscoverResponse:
    if req.provider not in rnk.PROVIDER_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider {req.provider!r}; choose one of {list(rnk.PROVIDER_CONFIGS)}",
        )
    api_key = _resolve_api_key(req.provider, req.api_key)

    job = Job(id=str(uuid.uuid4()), product=req.product, country=req.country, provider=req.provider)
    with _jobs_lock:
        _jobs[job.id] = job
    _executor.submit(_run_job, job, req, api_key)

    return DiscoverResponse(job_id=job.id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    job = _get_job(job_id)
    return JobStatusResponse(
        job_id=job.id, product=job.product, country=job.country, provider=job.provider,
        status=job.status, created_at=job.created_at, finished_at=job.finished_at,
        progress_tail=job.progress[-20:], error=job.error,
    )


@app.get("/jobs/{job_id}/results")
def job_results(job_id: str) -> list[dict]:
    job = _get_job(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"job {job_id!r} is {job.status}, not completed")
    return job.results or []


@app.get("/jobs")
def list_jobs() -> list[JobStatusResponse]:
    with _jobs_lock:
        jobs = list(_jobs.values())
    return [
        JobStatusResponse(
            job_id=j.id, product=j.product, country=j.country, provider=j.provider,
            status=j.status, created_at=j.created_at, finished_at=j.finished_at,
            progress_tail=j.progress[-5:], error=j.error,
        )
        for j in jobs
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
