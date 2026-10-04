"""What the host guarantees, whatever the model says: an inversion the model starts is followed
to its end before the model reads on; the answer's fixed parts (what was done, the parameters
used, the settings the gates changed) are written by code (paco.agent.answer). The host watches
no words, neither the user's nor the model's: what to do, and when to ask, the model decides."""

import json


def starts_inversion(name: str, arguments: str) -> bool:
    """Whether the call `name(arguments)` starts an inversion: invert, or redo of the inversion."""
    if name == "invert":
        return True
    if name != "redo":
        return False
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("stage") == "inversion"


def started_job(name: str, arguments: str, result: str) -> str | None:
    """The job the call `name(arguments)` started, from its `result`: an inversion's job_id;
    None when it started none."""
    if not starts_inversion(name, arguments):
        return None
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return None
    job_id = parsed.get("job_id") if isinstance(parsed, dict) else None
    return job_id if isinstance(job_id, str) and job_id else None


def job_running(status: str) -> bool:
    """Whether job_status's `status` is of a job still queued or running."""
    try:
        parsed = json.loads(status)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("state") in ("queued", "running")


def processed(result: str) -> str | None:
    """The run run_processing made, from its `result`; None when it made none (it gave the
    user's options instead, or failed)."""
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("options") or parsed.get("status") == "refused":
        return None
    run_id = parsed.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None
