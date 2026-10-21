"""The windows pick, decide and invert act on: those of the committed line a
spec names, by position, flag (any gate's, on the window's latest image, pick and model), G3's or
G5's verdict, or the model's decision. Resolved the same way each time."""

import json
import re
from collections.abc import Mapping
from pathlib import Path

from sigpipe.masw.runs import RunManifest

from paco.qc.log import latest, read_attempts
from paco.qc.models import Attempt, GateResult
from paco.qc.report import line_step

SPEC = (
    "all, xmids:a-b (m), xmids:x or xmids:[x, ...] (the windows nearest those positions), "
    "flag:<name>, verdict:<pass|retry|reject|none> (G3's; none: no curve), "
    "model:<pass|retry|reject|none> (G5's; none: no model) or status:<accepted|rejected|pending>, "
    "joined by + (each must hold)"
)
# How to name several windows, which a + does not: said when a spec does not read.
LISTED = (
    "a + keeps the windows every part names; to name several windows, list them, as "
    "xmids:[11.12, 12.38]"
)
_PART = re.compile(
    r"(all)|xmids:(-?[\d.]+)-(-?[\d.]+)|xmids:(\[.*\])|xmids:(-?[\d.]+)|flag:(\w+)"
    r"|verdict:(pass|retry|reject|none)|model:(pass|retry|reject|none)"
    r"|status:(accepted|rejected|pending)"
)


def select(run_folder: Path, spec: str, statuses: Mapping[str, str]) -> list[str]:
    """The windows of the run in `run_folder` that `spec` names (`SPEC`), in the line's order;
    `statuses`: each decided window's status (the others pending)."""
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    xmids = {window.folder: window.xmid for window in manifest.windows}
    attempts = read_attempts(run_folder)
    chosen = list(xmids)
    for text in (part.strip() for part in spec.split("+")):
        found = _PART.fullmatch(text)
        if found is None:
            raise ValueError(f"Unknown windows '{text}' in '{spec}': {SPEC}; {LISTED}.")
        every, low, high, listed, one, flag, verdict, model, status = found.groups()
        if every:
            continue
        if low is not None:
            chosen = [unit for unit in chosen if float(low) <= xmids[unit] <= float(high)]
        elif listed is not None or one is not None:
            positions = _positions(text, listed) if listed is not None else [float(one)]
            nearest = _nearest(xmids, positions)
            chosen = [unit for unit in chosen if unit in nearest]
        elif flag is not None:
            chosen = [unit for unit in chosen if flag in _flags(attempts, unit)]
        elif verdict is not None:
            chosen = [unit for unit in chosen if _verdict(attempts, unit) == verdict]
        elif model is not None:
            chosen = [unit for unit in chosen if _model(attempts, unit) == model]
        else:
            chosen = [unit for unit in chosen if statuses.get(unit, "pending") == status]
    if not chosen:
        raise ValueError(f"No window where {spec}" + ("; " + LISTED if "+" in spec else "") + ".")
    return chosen


def results_of(attempts: tuple[Attempt, ...], unit: str) -> list[GateResult]:
    """The window's latest results: G2's on its image, G3's and G4's on its pick, G5's and G6's
    on its model."""
    found: list[GateResult] = []
    for stage in ("phase_shift", "picking", "inversion"):
        attempt = latest(attempts, unit, stage)
        if attempt is not None:
            found.extend(attempt.results.values())
    return found


def _flags(attempts: tuple[Attempt, ...], unit: str) -> set[str]:
    return {flag.name for result in results_of(attempts, unit) for flag in result.flags}


def _verdict(attempts: tuple[Attempt, ...], unit: str) -> str:
    picked = latest(attempts, unit, "picking")
    g3 = picked.results.get("G3") if picked is not None else None
    return g3.verdict if g3 is not None else "none"


def _model(attempts: tuple[Attempt, ...], unit: str) -> str:
    inverted = latest(attempts, unit, "inversion")
    g5 = inverted.results.get("G5") if inverted is not None else None
    return g5.verdict if g5 is not None else "none"


def _positions(text: str, listed: str) -> list[float]:
    try:
        positions = [float(x) for x in json.loads(listed)]
    except ValueError, TypeError:
        raise ValueError(f"'{text}': xmids is a list of positions, m.") from None
    if not positions:
        raise ValueError(f"'{text}': at least one position.")
    return positions


def _nearest(xmids: Mapping[str, float], positions: list[float]) -> set[str]:
    """The window nearest each position; refused for a position farther than a line step from
    any window."""
    step = line_step(xmids.values())
    found: set[str] = set()
    for position in positions:
        unit = min(xmids, key=lambda one: abs(xmids[one] - position))
        if abs(xmids[unit] - position) > max(step, 1e-9):
            raise ValueError(f"No window at {position:g} m: the line's xmids run every {step:g} m.")
        found.add(unit)
    return found
