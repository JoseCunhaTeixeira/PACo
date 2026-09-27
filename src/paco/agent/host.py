"""What the host guarantees, whatever the model says: the parameters the stages ran with and
the settings the gates changed are printed after the answer, an answer that asks or offers once a stage tool has run is asked again once
(unless a tool said the agent is stuck), no inversion starts unless the user asked for models,
no petrophysical inversion unless they asked for soils or the water table, and an inversion the
model starts is followed to its end before the model reads on. A question before any stage tool
is the request's clarification, not an offer of more: asked again, the model may run what the
user did not ask for."""

import json
import re

# A tool said the request cannot be finished: the agent must ask the user.
STUCK = "you are stuck"

# The tools that do the request's work: an answer that asks or offers after one of them ran is
# asked again (before, the question is the request's clarification).
STAGE_TOOLS = frozenset({"run_processing", "pick", "invert", "redo", "job_status", "invert_petro"})

# The host's own turn, when an answer asks or offers without being stuck.
ANSWER_AGAIN = (
    "[PACo] Give your answer again, without any question or offer: the user will ask for more if "
    "they want it. Keep every result."
)

# Invert, inversion, inverse, inverser; a model, modèle; Vs; shear.
_MODELS = re.compile(r"\binver[ts]|\bmod[eè]les?\b|\bmodels?\b|\bvs\b|\bshear", re.IGNORECASE)
# Soils, sols; the water table, la nappe; N values, SPT; the soil types; petrophysics.
_SOILS = re.compile(
    r"\bsoils?\b|\bsols?\b|water[ -]table|\bnappe\b|\bN[ -]values?\b|\bSPT\b|\bclay|\bsand"
    r"|\bsilt|\bloam|\bargiles?\b|\bsables?\b|\blimons?\b|\bp[eé]tro",
    re.IGNORECASE,
)

_ASKS = re.compile(
    r"\?|\bchoose\b|\bwhich (one|option)\b|<options>|<offer>|\bwould you like\b|\bdo you want\b|"
    r"\bshall i\b|\blet me know\b|\bif you (would like|want|need|wish)\b",
    re.IGNORECASE,
)


def asks_for_models(question: str) -> bool:
    """Whether the user's message asks for velocity models (an inversion may run)."""
    return _MODELS.search(question) is not None


def asks_for_soils(question: str) -> bool:
    """Whether the user's message asks for soils or the water table (a petrophysical inversion
    may run)."""
    return _SOILS.search(question) is not None


def unasked(name: str, arguments: str, question: str) -> str | None:
    """Why the call `name(arguments)` is not made for `question`: an inversion the user did not
    ask for, said to the model; None when it may run."""
    if starts_inversion(name, arguments) and not asks_for_models(question):
        return (
            "Not called: the user asked for no velocity model, so no inversion starts. Answer "
            "with what the request asked for."
        )
    if name == "invert_petro" and not asks_for_soils(question):
        return (
            "Not called: the user asked for no soils or water table, so no petrophysical "
            "inversion starts. Answer with what the request asked for."
        )
    return None


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


def asks_or_offers(answer: str) -> bool:
    """Whether an answer asks the user something or offers more: a question, options to choose
    from, or "let me know", "would you like"."""
    return _ASKS.search(answer) is not None


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
