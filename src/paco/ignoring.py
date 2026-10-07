"""The settings a call gives that its tool does not have, set aside rather than refusing the call,
as pydantic ignores the fields a model does not expect: each name the tool does not know dropped
and said (the answer lists them, U6), the call run with the rest. A wrong setting still refuses
the call, with why, for the model to correct: a value the tool refuses for a setting it has, and
a name that is a setting's misspelling (`iterations` for `n_iterations`, `stack` for
`stacking`), the user's setting otherwise lost; what a call acts on (a profile, a run, a stage,
its windows or positions) is no setting."""

import difflib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.presets import (
    ActivePreset,
    PassivePreset,
    PresetError,
    apply_overrides,
    make_preset,
)

# Names a model gives the workers: PACo takes them from the user's message alone.
WORKERS = frozenset(
    {
        "workers",
        "n_workers",
        "num_workers",
        "max_workers",
        "n_jobs",
        "jobs",
        "cores",
        "n_cores",
        "cpus",
        "n_cpus",
        "threads",
        "n_threads",
        "processes",
        "n_processes",
    }
)
_SHOWN = 40  # characters of a value set aside, at most
# How close a name must be to a setting's to be its misspelling, refused rather than set aside:
# `iterations` (0.91 to n_iterations), `lenght` (0.83 to length); not `n_workers` (0.59 to
# n_layers), nor a pick's `mode` (0.62 to max_modes).
TYPO = 0.8
# The letters an abbreviation of a setting has at least, to be read as that setting's
# misspelling when it is the only one starting so (`stack` for stacking, 0.77 to it): fewer
# would start too many.
ABBREVIATION = 4
# The names a stage takes, as sigpipe's explanation lists them.
_ALLOWED = re.compile(r"Allowed: ([^.]+)\.")
# sigpipe's explanation of a setting its stage's method does not take.
_METHOD = re.compile(r"method '(\w+)' takes no parameters")


@dataclass(frozen=True)
class Cleaned:
    """A call's settings without the names its tool does not have, and each one set aside, in
    words: "n_workers 10: not an inversion setting"."""

    values: dict[str, Any] | None
    ignored: tuple[str, ...] = ()
    # The names that misspell a setting, kept for the tool to refuse, each with the setting.
    typos: tuple[str, ...] = ()


def preset_overrides(
    base: str | ActivePreset | PassivePreset, overrides: Mapping[str, Any] | None
) -> Cleaned:
    """`overrides` without the stages, and the settings of a stage, that the preset does not
    have: on a new preset of mode `base` (run_processing's, compare's variants), or over a run's
    own preset (a redo of the records or the images). Values stay as given, the tool judging
    them; the mode and the windows in metres pass as they are (the tool reads them)."""
    if not overrides:
        return Cleaned(dict(overrides) if overrides is not None else None)

    def unknown(values: Mapping[str, Any]) -> str | None:
        """Why the preset refuses `values`, when it is only for names it does not have (its
        explanation's line: "unknown parameter; method 'none' takes no parameters"); None when
        it takes them, or refuses them for another reason."""
        try:
            if isinstance(base, str):
                make_preset(base, values)
            else:
                apply_overrides(base, values)
        except PresetError as error:
            cause = error.__cause__
            if isinstance(cause, ValidationError) and all(
                one["type"] == "extra_forbidden" for one in cause.errors()
            ):
                lines = [line.strip("- ") for line in str(error).splitlines()[1:]]
                return lines[0].split(": ", 1)[-1] if lines else "unknown"
        return None

    kept: dict[str, Any] = {}
    ignored: list[str] = []
    for stage, values in overrides.items():
        if stage == "mode":
            kept[stage] = values
        elif (why := unknown({stage: {} if isinstance(values, Mapping) else values})) is not None:
            if _typo(stage, _allowed(why)) is not None:
                kept[stage] = values  # a stage misspelled: sigpipe says which it is
            else:
                ignored.append(_said(stage, values, _for(stage, "not a processing stage")))
        elif isinstance(values, Mapping):
            for key, value in cast(Mapping[str, Any], values).items():
                metres = stage == "masw" and key in ("length_m", "step_m")
                why = None if metres else unknown({stage: {key: value}})
                if why is None or _typo(key, _allowed(why)) is not None:
                    # Taken, or a setting misspelled: sigpipe says which it is.
                    kept.setdefault(stage, {})[key] = value
                    continue
                # A setting its stage has under another method: that method's, said.
                method = _METHOD.search(why)
                reason = (
                    f"not a {stage} setting with method {method.group(1)}"
                    if method
                    else f"not a {stage} setting"
                )
                ignored.append(_said(f"{stage} {key}", value, _for(key, reason)))
        else:
            kept[stage] = values
    return Cleaned(kept or None, tuple(ignored))


def picking_changes(changes: Mapping[str, Any] | None) -> Cleaned:
    """`changes` of the picking without the names it does not have; a name misspelling a
    picking setting kept out and said as a typo, for the tool to refuse."""
    if not changes:
        return Cleaned(dict(changes) if changes is not None else None)
    known = list(PickingParameters.model_fields)
    ignored: list[str] = []
    typos: list[str] = []
    kept: dict[str, Any] = {}
    for name, value in changes.items():
        if name in known:
            kept[name] = value
        elif (meant := _typo(name, known)) is not None:
            typos.append(f"{name}: not a picking setting. Did you mean {meant}?")
        else:
            why = (
                "not a picking setting: PACo picks M0, a higher mode is picked by hand in "
                "PAC's Dispersion picking page"
                if name in ("mode", "modes")
                else _for(name, "not a picking setting")
            )
            ignored.append(_said(name, value, why))
    return Cleaned(kept or None, tuple(ignored), tuple(typos))


def inversion_parameters(parameters: Mapping[str, Any] | None) -> Cleaned:
    """`parameters` of the inversion without the names it does not have; a name misspelling one
    kept, for the inversion's check to refuse (saying which it is), and said as a typo."""
    if not parameters:
        return Cleaned(dict(parameters) if parameters is not None else None)
    known = list(InversionParameters.model_fields)
    ignored: list[str] = []
    typos: list[str] = []
    kept: dict[str, Any] = {}
    for name, value in parameters.items():
        if name in known:
            kept[name] = value
        elif (meant := _typo(name, known)) is not None:
            kept[name] = value
            typos.append(f"{name}: not an inversion setting. Did you mean {meant}?")
        else:
            ignored.append(_said(name, value, _for(name, "not an inversion setting")))
    return Cleaned(kept or None, tuple(ignored), tuple(typos))


def undeclared(arguments: Mapping[str, Any], declared: Mapping[str, Any], tool: str) -> Cleaned:
    """A call's `arguments` without those its tool does not declare (`declared`, its input
    schema's properties), which the server would ignore unsaid."""
    ignored = tuple(
        _said(name, value, _for(name, f"not an argument of {tool}"))
        for name, value in arguments.items()
        if name not in declared
    )
    kept = {name: value for name, value in arguments.items() if name in declared}
    return Cleaned(kept, ignored)


def _typo(name: str, known: Iterable[str]) -> str | None:
    """The setting `name` misspells, when it is that close to one (TYPO) or abbreviates only
    it (ABBREVIATION); never the workers' names, which PACo takes from the user's message
    alone."""
    if name.lower() in WORKERS:
        return None
    known = list(known)
    # Cut within a word (stack|ing), not a whole one (a pick's `mode`, not mode_min_ratio).
    starting = [one for one in known if one.startswith(name) and one[len(name) :][:1].isalpha()]
    if len(name) >= ABBREVIATION and len(starting) == 1:
        return starting[0]
    matches = difflib.get_close_matches(name, known, n=1, cutoff=TYPO)
    return matches[0] if matches else None


def _allowed(why: str) -> list[str]:
    """The names sigpipe's explanation `why` lists as allowed."""
    found = _ALLOWED.search(why)
    return [name.strip() for name in found.group(1).split(",")] if found else []


def _for(name: str, reason: str) -> str:
    """`reason`, with the workers' own when `name` gives the workers."""
    if name.lower() in WORKERS:
        return f"{reason}: the workers are your message's, which PACo gives every stage"
    return reason


def _said(name: str, value: object, reason: str) -> str:
    shown = json.dumps(value, default=str)
    if len(shown) > _SHOWN:
        shown = shown[: _SHOWN - 1] + "…"
    return f"{name} {shown}: {reason}"
