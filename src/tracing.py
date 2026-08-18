"""
Lightweight structured tracing for the pipeline's phase boundaries
(Discovery -> Scrape -> Directory/Report Mining -> Ranking -> Validation).

Emits one JSON line per phase start/end to stderr -- easy to grep, pipe into
`jq`, or feed a log aggregator, without pulling in the OpenTelemetry SDK for
a single-process pipeline this size. Existing print()/on_progress per-item
progress lines throughout src/ are unchanged; this only wraps the five phase
boundaries with a start/end/duration/status record.

Usage:
    with tracing.span("discovery", product=product, country=country):
        candidates = disc.discover(...)
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


def _emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, ensure_ascii=False, default=str), file=sys.stderr, flush=True)


@contextmanager
def span(phase: str, **attributes: Any) -> Iterator[None]:
    """Context manager wrapping one pipeline phase. Emits a phase_start
    record on entry and a phase_end record (with duration_seconds and
    status "ok"/"error") on exit, re-raising any exception unchanged."""
    start = time.monotonic()
    _emit({
        "event": "phase_start",
        "phase": phase,
        "ts": datetime.now(timezone.utc).isoformat(),
        **attributes,
    })
    status = "ok"
    error: str | None = None
    try:
        yield
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        record = {
            "event": "phase_end",
            "phase": phase,
            "ts": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - start, 3),
            "status": status,
            **attributes,
        }
        if error is not None:
            record["error"] = error
        _emit(record)
