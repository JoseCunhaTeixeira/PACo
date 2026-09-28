"""A shot's time origin as a record's preprocessing left it: the trigger is part of the muting
(the user, 2026-09-28). Where a record's first breaks should put the shot, whether a window's
records were muted, and each record's pulse, the width its mute keeps after the slowest
arrival."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import BaseModel
from sigpipe.masw.profiles import Profile
from sigpipe.masw.windows import MASWWindow

from paco.qc.loops import deep_merge
from paco.qc.models import Attempt


def file_triggers(profile: Profile) -> dict[str, float | None]:
    """Each record's trigger, s after its first sample, from its file; None when it says none."""
    return {record.path.name: record.trigger_s for record in profile.records}


def preprocessing_values(preset: BaseModel, attempt: Attempt | None) -> dict[str, Any]:
    """What a record's preprocessing had at `attempt`: the run's preset with its changes."""
    base = preset.model_dump(mode="json")
    return deep_merge(base, attempt.parameters) if attempt is not None else base


def trigger_context(
    values: Mapping[str, Any], file_trigger_s: float | None
) -> tuple[float, float | None]:
    """Where the first breaks should put the shot on the preprocessed record, and the shift its
    muting applied (None: off). On, the time origin moved by the trigger's t0 or, left out, by
    the file's own: the shot then at 0. Off, the record as recorded: the shot where its file
    says, 0 when it says nothing."""
    trigger = values.get("trigger")  # none on a passive line, nor muting
    if trigger is None or values["muting"]["method"] != "mute":
        return file_trigger_s or 0.0, None
    t0 = trigger.get("t0")
    return 0.0, float(t0 if t0 is not None else file_trigger_s or 0.0)


def window_muted(window_folder: Path, muted: Callable[[str], bool]) -> bool:
    """Whether a record of the window in `window_folder` was muted (`muted`, by record name)."""
    window = MASWWindow.model_validate_json((window_folder / "window.json").read_text())
    return any(muted(path.name) for path in window.selected_files)


def muted_records(
    preset: BaseModel, attempts: Iterable[Attempt], latest: Callable[..., Attempt | None]
) -> Callable[[str], bool]:
    """Whether a record, by name, was muted at its latest preprocessing."""
    listed = tuple(attempts)

    def muted(name: str) -> bool:
        values = preprocessing_values(preset, latest(listed, name, "preprocessing"))
        return values.get("muting", {}).get("method") == "mute"  # never on a passive line

    return muted


def pulse_widths(
    attempts: Iterable[Attempt], names: Iterable[str], fallback: float
) -> dict[str, float]:
    """The width each record's mute keeps after the slowest arrival (the user, 2026-09-28): its
    own pulse as G1 measured it on the record not muted, else the line's median pulse, else
    `fallback` when no record's could be measured."""
    measured: dict[str, float] = {}
    for attempt in attempts:  # in the order logged: the latest measure wins
        g1 = attempt.results.get("G1") if attempt.stage == "preprocessing" else None
        pulse = next((m.value for m in g1.metrics if m.name == "pulse_s"), None) if g1 else None
        if pulse is not None:
            measured[attempt.unit] = float(pulse)
    line = median(measured.values()) if measured else fallback
    return {name: measured.get(name, line) for name in names}


def with_pulse(parameters: dict[str, Any], width: float) -> dict[str, Any]:
    """`parameters` with the width of a mute that gives none: `width`, the record's pulse. A
    width given (by the user) stays."""
    muting = parameters.get("muting")
    if not isinstance(muting, dict) or muting.get("method") != "mute" or "width" in muting:
        return parameters
    return {**parameters, "muting": {**muting, "width": round(width, 4)}}
