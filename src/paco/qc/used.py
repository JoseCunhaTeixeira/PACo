"""What a stage ran with, and why, in words, for the user: the window length and the phase
shift's band of a run, the picker's settings, the inversions' layers, bounds and effort, each
with where it comes from (given, a rule on the data, a gate, or a default). The host lists them
after every answer, as it lists the settings the gates changed."""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.runs import RunManifest

from paco.qc.models import Attempt

GIVEN = "given (in your request, or chosen by the agent)"
# The ladder's note (coherence.choose_length): why its length, and each length's trials.
_LADDER = re.compile(r"masw length \d+ for the whole line, (.*?): trial windows G3 passed ")
_TRIAL = re.compile(r"(\d+/\d+) at (\d+)(?: \(([^)]*)\))?")
DEFAULT = "the preset's default"


def processing_used(
    manifest: RunManifest, given: Mapping[str, Any] | None = None, notes: Sequence[str] = ()
) -> tuple[str, ...]:
    """The run's mode, its MASW windows and the settings of each stage, as run.json has them,
    each with its reason: `given` (the overrides of the call), the rules' `notes` on the line
    (the window length's trials, the shots' reach, the band's cap), or the preset."""
    preset: dict[str, Any] = manifest.preset.model_dump(mode="json")
    masw = preset.pop("masw")
    spacing = manifest.profile.receiver_spacing_m
    given = given or {}

    def asked(stage: str, key: str) -> bool:
        values = given.get(stage)
        return isinstance(values, Mapping) and key in cast(Mapping[str, Any], values)

    def rule(prefix: str) -> str | None:
        note = next((note for note in notes if note.startswith(prefix)), None)
        return note.split(": ", 1)[1].rstrip(".") if note is not None else None

    xmids = [window.xmid for window in manifest.windows]
    where = f", {len(xmids)} windows, xmid {min(xmids):g} to {max(xmids):g} m" if xmids else ""
    length = masw["length"]
    ladder = next((note for note in notes if note.startswith("masw length")), None)
    if asked("masw", "length"):
        length_why = GIVEN + (f"; its trial windows: {_tried(ladder)}" if ladder else "")
    elif ladder:
        length_why = f"{_ladder_rule(ladder)}; trial windows G3 passed: {_tried(ladder)}"
    else:
        length_why = DEFAULT
    used = [
        f"mode {preset.pop('mode')}: " + (GIVEN if "mode" in given else "the profile's kind"),
        f"MASW windows of {length} receivers ({(length - 1) * spacing:g} m){where}: {length_why}",
        f"windows every {masw['step']} receiver{'s' if masw['step'] != 1 else ''} "
        f"({masw['step'] * spacing:g} m): " + (GIVEN if asked("masw", "step") else DEFAULT),
        f"shots {_distances(masw['distance_min'], masw['distance_max'])} from a window's middle: "
        + _shots_why(
            GIVEN if asked("masw", "distance_min") else rule("near_field distance_m"),
            GIVEN if asked("masw", "distance_max") else rule("masw distance_max"),
        ),
    ]
    dispersion = preset.pop("dispersion", None)
    if isinstance(dispersion, Mapping):
        values = cast(Mapping[str, Any], dispersion)
        band = rule("dispersion fmax") or rule("dispersion fmin")
        used.append(
            f"phase shift {_value(values, 'fmin')} to {_value(values, 'fmax')} Hz: "
            + (
                GIVEN
                if asked("dispersion", "fmin") or asked("dispersion", "fmax")
                else band or DEFAULT
            )
        )
        used.append(
            f"velocities {_value(values, 'vmin')} to {_value(values, 'vmax')} m/s"
            + (f" ({values['nv']} steps)" if "nv" in values else "")
            + ": "
            + (GIVEN if asked("dispersion", "vmin") or asked("dispersion", "vmax") else DEFAULT)
        )
    muting = preset.pop("muting", None)
    if isinstance(muting, Mapping):
        values = cast(Mapping[str, Any], muting)
        if values.get("method") not in (None, "none"):
            bounds = " to ".join(
                f"{values[key]:g}" for key in ("vmin", "vmax") if values.get(key) is not None
            )
            used.append(
                f"muting {bounds} m/s: "
                + (GIVEN if "muting" in given else rule("muting mute") or DEFAULT)
            )
    for stage, values in preset.items():
        if isinstance(values, Mapping):
            method = cast(Mapping[str, Any], values).get("method")
            if method not in (None, "none"):
                used.append(f"{stage} {method}: " + (GIVEN if stage in given else DEFAULT))
    return tuple(used)


def picking_used(
    parameters: PickingParameters, given: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """The picker's settings every window started from: PACo's, which G3 changes window by
    window (in the gates' changes); then the values the call gave (`given`), if any."""
    band = ""
    if parameters.fmin is not None or parameters.fmax is not None:
        band = f", {parameters.fmin or 0:g} to {parameters.fmax or 'the image'} Hz"
    return (
        f"picking: M0 tracked along its ridge (threshold {parameters.threshold:g}, corridor "
        f"{parameters.corridor:.0%} of the velocity{band}), points under "
        f"{parameters.min_relative_coherence:g} of the mode's coherence dropped, resampled every "
        f"{parameters.wavelength_step:g} m of wavelength: PACo's starting values, changed "
        "window by window where G3 asked",
        *(
            [
                "picking "
                + ", ".join(f"{name} {_shown_given(value)}" for name, value in given.items())
                + ": given (in your request, or chosen by the agent)"
            ]
            if given
            else []
        ),
    )


def _shown_given(value: Any) -> str:  # noqa: ANN401
    return f"{value:g}" if isinstance(value, float) else json.dumps(value)


def inversion_used(
    attempts: Iterable[Attempt], units: Sequence[str], given: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """The parameters the windows `units` were last inverted with, as ranges over the windows,
    each with its rule: layers, Vs bounds, the half-space's depth, the sampler's effort. `given`
    are the values the user gave. Each window's own are in its
    SeismicInversion_Parameters_0000.json."""
    latest: dict[str, Mapping[str, Any]] = {}
    for attempt in attempts:
        if attempt.stage == "inversion" and attempt.unit in units and attempt.parameters:
            latest[attempt.unit] = attempt.parameters
    if not latest:
        return ()
    given = given or {}
    runs = list(latest.values())
    # The layers chosen by the data, or given (an older log names no layering).
    free = [
        run
        for run in runs
        if run.get("layering", "fixed" if "vs_layers" in run else "free") == "free"
    ]
    fixed = [run for run in runs if run not in free]
    said: list[str] = []
    if free:
        bounds = [run["free"] for run in free]
        asked = given.get("free") or {}
        said += [
            f"inversion of {len(free)} windows, the layers chosen by the data: up to "
            f"{_span(bound['max_layers'] for bound in bounds)} layers, the half-space among them: "
            + (
                GIVEN
                if "max_layers" in asked
                else "8 at first, more where the models piled at the most allowed (G5)"
            ),
            f"Vs from {_span(bound['vs_min'] for bound in bounds)} to "
            f"{_span(bound['vs_max'] for bound in bounds)} m/s: "
            + (
                GIVEN + ", checked against each curve"
                if "vs_min" in asked or "vs_max" in asked
                else "100 to 2,000 m/s, wider where a curve needed it or the models piled at a "
                "bound (G5)"
            ),
            f"interfaces from {_span(bound['depth_min'] for bound in bounds)} to "
            f"{_span(bound['depth_max'] for bound in bounds)} m deep: a third of each curve's "
            "shortest wavelength (thinner is not resolved) to half its longest (deeper is not "
            "resolved)",
        ]
    if fixed:
        above = [layer for run in fixed for layer in run["vs_layers"][:-1]]
        half_spaces = [run["vs_layers"][-1] for run in fixed]
        depths = [sum(layer["thickness_max"] for layer in run["thickness_layers"]) for run in fixed]
        thinnest = [layer["thickness_min"] for run in fixed for layer in run["thickness_layers"]]
        said += [
            f"inversion of {len(fixed)} windows, the layers given: "
            f"{_span(run['n_layers'] for run in fixed)} layers, the half-space among them: "
            + (
                GIVEN
                if "n_layers" in given
                else "4 to start, never fewer than 3; then one more where a model misfit, one "
                "fewer where a layer was not resolved or two were alike (G5)"
            ),
            f"Vs bounds by window: lowest {_span(layer['vs_min'] for layer in above)}, highest "
            f"{_span(layer['vs_max'] for layer in above)} m/s; the half-space's highest "
            f"{_span(layer['vs_max'] for layer in half_spaces)} m/s: "
            + (
                GIVEN + ", checked against each curve"
                if "vs_layers" in given
                else "wide at first (100 to 1,000 m/s, the half-space to 2,000), wider where a "
                "curve needed it or a posterior piled at a bound (G5)"
            ),
            f"layers at least {_span(thinnest)} m thick: a third of each curve's shortest "
            "wavelength, thinner is not resolved",
            f"the half-space's top at most {_span(depths)} m deep: half each curve's longest "
            "wavelength at first, shrunk to the depth the data inform where it was shallower "
            "(G5)",
        ]
    drops = [100 * float(run.get("max_vs_drop", 1.0)) for run in runs]
    said += [
        f"a layer's Vs at most {_span(drops)} % below the one above: "
        + (
            GIVEN
            if "max_vs_drop" in given
            else "a fifth; under a layer much stiffer than the next, the forward model's "
            "fundamental mode is a wave trapped in the soft layer"
        ),
        f"{_span(run['n_iterations'] for run in runs)} iterations "
        f"({_span(run['n_burnin_iterations'] for run in runs)} burn-in), "
        f"{_span(run['n_chains'] for run in runs)} chains: "
        + (
            GIVEN
            if "n_iterations" in given
            else "PAC's effort, longer where the chains did not agree (G5)"
        ),
        "each window's own: SeismicInversion_Parameters_0000.json in its folder",
    ]
    return tuple(said)


def _distances(near: float | None, far: float | None) -> str:
    """The distances a window's shots lie within, a bound left out none: "2 to 30 m", "any
    distance"."""
    if near is None and far is None:
        return "at any distance"
    if far is None:
        return f"beyond {near:g} m"
    return f"within {far:g} m" if near is None else f"{near:g} to {far:g} m"


def _shots_why(near: str | None, far: str | None) -> str:
    """Why a window's shots lie between its two distances: each limit's reason, or the
    preset's."""
    said = [f"nearest, {near}" if near else "", f"farthest, {far}" if far else ""]
    return "; ".join(part for part in said if part) or DEFAULT


def _ladder_rule(note: str) -> str:
    """Why the ladder kept its length, from its note ("masw length 11 for the whole line, the
    most precise that passed (none within 20%): ...")."""
    found = _LADDER.match(note)
    return f"the ladder's choice, {found.group(1)}" if found else "the ladder's choice"


def _tried(note: str) -> str:
    """The trials of the length note ("...: trial windows G3 passed 25/27 at 5 (picks 40%),
    14/27 at 16 (compared)"), said per length."""
    said: list[str] = []
    for passed, length, extra in _TRIAL.findall(note.rpartition(" G3 passed ")[2]):
        parts = [
            "tried for comparison" if part == "compared" else part
            for part in (extra.split("; ") if extra else [])
        ]
        said.append(f"{passed} at {length} receivers" + (f" ({', '.join(parts)})" if parts else ""))
    return ", ".join(said)


def _span(values: Iterable[float]) -> str:
    """The values' range as said: "4", or "3-5"."""
    ordered = sorted({round(float(value), 2) for value in values})
    if not ordered:
        return "n/a"
    low, high = (_number(value) for value in (ordered[0], ordered[-1]))
    return low if low == high else f"{low}-{high}"


def _number(value: float) -> str:
    return f"{int(value):,}" if value.is_integer() else f"{value:g}"


def _value(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    return "the profile's" if value is None else f"{value:g}"
