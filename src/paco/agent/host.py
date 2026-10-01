"""What the host guarantees, whatever the model says: the parameters the stages ran with and
the settings the gates changed are printed after the answer, and an inversion the model starts
is followed to its end before the model reads on. The host watches no words, neither the user's
nor the model's: what to do, and when to ask, the model decides."""

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


def changed_items(result: str) -> list[str]:
    """The `changed` list of a tool's result: the settings the gates and the checks changed."""
    return _items(result, "changed")


def used_items(result: str) -> list[str]:
    """The `used` list of a tool's result: the parameters the stage ran with."""
    return _items(result, "used")


def with_changes(answer: str, changes: list[str], used: list[str] | None = None) -> str:
    """`answer`, with the parameters the stages ran with and the settings the gates changed
    listed after it, as the tools gave them."""
    blocks = [answer.rstrip()]
    for title, items in (("Parameters used", used or []), ("Settings the gates changed", changes)):
        if items:
            blocks.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(blocks) if len(blocks) > 1 else answer


def _items(result: str, key: str) -> list[str]:
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return []
    items = parsed.get(key) if isinstance(parsed, dict) else None
    return [str(item) for item in items] if isinstance(items, list) else []
