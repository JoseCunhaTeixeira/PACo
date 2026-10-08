"""The settings the user gave for a run, locked for it (U2, L6): the gates and the rules never
change them; a gate that needs one changed rejects its window instead, saying the change it
asks, for the agent to suggest. A limit the user gave (the picking's band) bounds the gates'
changes instead: they may narrow the range within it, never take it past. Kept in the run's QC
log, as events (S2): each call's values, by family (the processing's overrides, preset paths;
the picking's changes; the inversion's parameters), the latest value of a setting winning."""

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from sigpipe.masw.runs.history import LOG_FILE, LOG_VERSION, log_entries, log_lock

from paco.qc.log import made_by_now
from paco.qc.models import Stage

# The log's event of the settings a call gave.
GIVEN = "given"
type Family = Literal["processing", "picking", "inversion"]
# Which family a stage's settings belong to (the petrophysical inversion takes none).
FAMILY: dict[Stage, Family] = {
    "preprocessing": "processing",
    "phase_shift": "processing",
    "picking": "picking",
    "inversion": "inversion",
}
# What the run's settings are, not values: never locked (the processing's mode).
_NOT_SETTINGS = frozenset({"mode"})
# The settings the user gives as a limit, not a value, by family, each with the value a gate's
# change takes within it: the picking's band (down to fmin, up to fmax), which a gate may
# narrow (a higher fmin, a lower fmax: an alias, competing ridges), never widen. The image's
# band (the processing's dispersion fmin and fmax) is what the user asked computed: a value.
type Limit = Callable[[float, float], float]
LIMITS: dict[Family, dict[str, Limit]] = {"picking": {"fmin": max, "fmax": min}}


def give(run_folder: Path, family: Family, values: Mapping[str, Any] | None) -> None:
    """Lock `values`, the user's for `family`, in run `run_folder`: an event of the run's log,
    logged by the agent's call that passed them, merged over those given before (the latest
    value of a setting wins)."""
    kept = {key: value for key, value in (values or {}).items() if key not in _NOT_SETTINGS}
    if not kept:
        return
    event = {
        "event": GIVEN,
        "version": LOG_VERSION,
        "family": family,
        "values": kept,
        "at": datetime.now(UTC).isoformat(),
        "actor": "agent",
        "made_by": made_by_now("agent").model_dump(mode="json"),
    }
    with log_lock(run_folder), (run_folder / LOG_FILE).open("a") as log:
        log.write(json.dumps(event, default=str) + "\n")


def given_of(run_folder: Path) -> dict[str, dict[str, Any]]:
    """The settings the user gave for the run, by family (none: {}): the log's events merged
    in order."""
    given: dict[str, dict[str, Any]] = {}
    for entry in log_entries(run_folder):
        family, values = entry.get("family"), entry.get("values")
        if entry.get("event") == GIVEN and isinstance(family, str) and isinstance(values, dict):
            given[family] = _merged(given.get(family, {}), cast(Mapping[str, Any], values))
    return given


def locked(run_folder: Path, stage: Stage) -> dict[str, Any]:
    """The settings the user gave that `stage`'s changes may not touch."""
    family = FAMILY.get(stage)
    return given_of(run_folder).get(family, {}) if family is not None else {}


def limits_of(stage: Stage) -> dict[str, Limit]:
    """The limits among the settings of `stage`'s family (LIMITS)."""
    family = FAMILY.get(stage)
    return LIMITS.get(family, {}) if family is not None else {}


def unlocked(
    changes: Mapping[str, Any],
    given: Mapping[str, Any],
    limits: Mapping[str, Limit] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """`changes` without the leaves the user gave, and those leaves (the changes held back); a
    change of a limit the user gave (`limits`, the family's own settings) taken within it."""
    free: dict[str, Any] = {}
    held: dict[str, Any] = {}
    for key, value in changes.items():
        mine = given.get(key)
        if isinstance(value, Mapping) and isinstance(mine, Mapping):
            inner_free, inner_held = unlocked(
                cast(Mapping[str, Any], value), cast(Mapping[str, Any], mine)
            )
            if inner_free:
                free[key] = inner_free
            if inner_held:
                held[key] = inner_held
        elif (within := _within(key, value, mine, limits)) is not None:
            free[key] = within
        elif key in given:
            held[key] = value
        else:
            free[key] = value
    return free, held


def beyond(
    changes: Mapping[str, Any], given: Mapping[str, Any], limits: Mapping[str, Limit]
) -> dict[str, Any]:
    """The changes of `changes` past a limit the user gave, as asked (fmax 60 past their 50),
    which `unlocked` takes within it."""
    return {
        key: value
        for key, value in changes.items()
        if (within := _within(key, value, given.get(key), limits)) is not None and within != value
    }


def said(held: Mapping[str, Any], given: Mapping[str, Any], path: tuple[str, ...] = ()) -> str:
    """The changes held back, in words: "dispersion vmax 900 (given: 400)"."""
    parts: list[str] = []
    for key, value in held.items():
        mine = given.get(key)
        if isinstance(value, Mapping) and isinstance(mine, Mapping):
            parts.append(
                said(cast(Mapping[str, Any], value), cast(Mapping[str, Any], mine), (*path, key))
            )
        else:
            parts.append(f"{' '.join((*path, key))} {_shown(value)} (given: {_shown(mine)})")
    return "; ".join(part for part in parts if part)


def leaves(values: Mapping[str, Any], path: tuple[str, ...] = ()) -> list[str]:
    """`values` in words, a leaf each: "dispersion vmax 400"."""
    found: list[str] = []
    for key, value in values.items():
        if isinstance(value, Mapping):
            found += leaves(cast(Mapping[str, Any], value), (*path, key))
        else:
            found.append(f"{' '.join((*path, key))} {_shown(value)}")
    return found


def _merged(base: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in new.items():
        old = merged.get(key)
        if isinstance(old, Mapping) and isinstance(value, Mapping):
            merged[key] = _merged(cast(Mapping[str, Any], old), cast(Mapping[str, Any], value))
        else:
            merged[key] = value
    return merged


def _within(
    key: str,
    value: Any,  # noqa: ANN401
    mine: Any,  # noqa: ANN401
    limits: Mapping[str, Limit] | None,
) -> float | None:
    """`value`, a change of `key`, taken within the limit the user gave it (`mine`); None when
    `key` is no limit they gave, or `value` no frequency (a band removed: held, as given)."""
    limit = (limits or {}).get(key)
    numbers = all(
        isinstance(one, int | float) and not isinstance(one, bool) for one in (value, mine)
    )
    return float(limit(value, mine)) if limit is not None and numbers else None


def _shown(value: Any) -> str:  # noqa: ANN401
    return f"{value:g}" if isinstance(value, float) else json.dumps(value)
