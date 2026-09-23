"""What the agent reads of an inversion job: progress, and the models found so far."""

from datetime import UTC, datetime
from itertools import accumulate

from paco.inversion.models import InversionRecord, InversionStatus

_MAX_ERRORS = 3
_ERROR_LENGTH = 200


def summarize_inversion(record: InversionRecord, live: bool) -> InversionStatus:
    """The status of job `record`; `live` says whether it still runs in this server.

    A job left queued or running by a server that stopped is reported as interrupted.
    """
    state, error = record.state, record.error
    if state in ("queued", "running") and not live:
        state = "interrupted"
        error = "The server stopped before the job finished: call invert again."

    succeeded = [window for window in record.windows if window.status == "succeeded"]
    vs = [window.vs_m_s for window in succeeded if window.vs_m_s is not None]
    depths = [
        tuple(accumulate(window.thicknesses_m))
        for window in succeeded
        if window.thicknesses_m is not None
    ]
    elapsed_s = None
    if record.started_at is not None:
        end = record.finished_at or datetime.now(UTC)
        elapsed_s = round((end - record.started_at).total_seconds(), 1)

    return InversionStatus(
        job_id=record.job_id,
        run_id=record.run_id,
        state=state,
        done=len(record.windows),
        total=record.total,
        n_failed=len(record.windows) - len(succeeded),
        elapsed_s=elapsed_s,
        vs_m_s=_ranges(vs),
        depths_m=_ranges(depths),
        errors=_distinct_errors(record),
        error=error,
    )


def _ranges(values: list[tuple[float, ...]]) -> tuple[tuple[float, float], ...]:
    """Per position in the tuples (a layer, an interface), the rounded range across windows."""
    return tuple(
        (round(min(column), 1), round(max(column), 1)) for column in zip(*values, strict=True)
    )


def _distinct_errors(record: InversionRecord) -> tuple[str, ...]:
    errors: list[str] = []
    seen: set[str] = set()
    for window in record.windows:
        if window.error is None or window.error in seen:
            continue
        seen.add(window.error)
        errors.append(f"xmid {window.xmid:.2f}: {window.error}"[:_ERROR_LENGTH])
        if len(errors) == _MAX_ERRORS:
            break
    return tuple(errors)
