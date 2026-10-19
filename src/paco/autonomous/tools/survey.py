"""survey (docs/autonomous_plan.md §4.1): what the line is before any setting is chosen. Its geometry and
the limits it sets the windows and the images; its raw records as G1 measures them (no mute, no
filter: the preset's defaults), with the line's reach and the surface waves' cone from the gathers
of its shots. A passive line's records are noise, without shot: no reach nor cone, its images'
frequency step its segments'. The modes the line takes. Made once a study (survey.json): P2's tools
open once it has run."""

import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sigpipe.masw.presets import make_preset, resolve_preset
from sigpipe.masw.profiles import MODES, load_profile
from sigpipe.masw.runs.writing import write_atomic

from paco.autonomous.tools.trials import SURVEY_FILE, Study, settled, with_records
from paco.qc.coherence import receiver_spacing
from paco.qc.config import QCConfig
from paco.qc.g4_profile import LINE
from paco.qc.models import GateResult
from paco.qc.muting import estimate_cone, mute_candidates
from paco.settings import Settings

SURVEY = "survey"  # the study's folder of the raw records' G1 log
SAVED = SURVEY_FILE
# The shots, or the flagged records, said one by one at most; past it, summed up.
MANY = 8
# The modes, as the survey says them.
MODE_SAID = {
    "active": "each window stacks its shots' images",
    "passive-active": "each window correlates its farther shots, as noise, and stacks them",
    "passive": "each window correlates the noise's segments and stacks them",
}
# The window lengths the plan allows, in receivers: at least 5 (a phase-shift image needs them),
# at most half the line (longer ones lose the lateral detail it was shot for), §4.2.
SHORTEST_WINDOW = 5


class Shot(BaseModel):
    """A record's shot: where it stands against the line."""

    model_config = ConfigDict(frozen=True)

    record: str
    x_m: float | None  # None when its file does not say
    side: Literal["before the line", "after the line", "on the line", "unknown"]
    offset_m: float | None  # from the line's nearest end; 0 on the line
    trigger_s: float | None  # the shot's delay its file states


class Survey(BaseModel):
    """The line, its limits, and its raw records."""

    model_config = ConfigDict(frozen=True)

    profile: str
    receivers: int
    spacing_m: float
    line_m: float  # from the first receiver to the last
    shots: tuple[Shot, ...]
    sampling_hz: float
    nyquist_hz: float
    record_s: tuple[float, float]  # the shortest record and the longest
    frequency_step_hz: float  # 1/T of the shortest record: an image's resolution in frequency
    shortest_wavelength_m: float  # 2·dx: shorter, a ridge may be an alias
    lengths: tuple[int, int]  # the window lengths allowed, receivers
    reach_m: float | None  # beyond it from the shot, the traces' median SNR falls under G1's limit
    cone_m_s: tuple[float, float, float] | None  # the surface waves' slow edge, median, fast edge
    cone_traces: int | None
    usable_hz: tuple[float, float] | None  # the records' median usable band
    usable_spread_hz: tuple[float, float] | None  # the narrowest low edge and high edge
    g1: dict[str, GateResult]  # by record, and on the line
    # A passive line's records are ambient noise; the modes the line takes, its own first; a
    # passive line's segments (the preset's), its images' frequency step their inverse.
    kind: Literal["active", "passive"] = "active"
    modes: tuple[str, ...] = ("active", "passive-active")
    segment_s: float | None = None


def survey(study: Study, settings: Settings, config: QCConfig) -> Survey:
    """The study's survey of its line: made the first time, read after."""
    saved = study.folder / SAVED
    if saved.exists():
        return Survey.model_validate_json(saved.read_text())
    profile = load_profile(study.profile, settings)
    preset = resolve_preset(make_preset(profile.kind), profile)
    passive = profile.kind == "passive"
    # The surface waves' cone from the shots' gathers, a passive line has none.
    cone = None if passive else estimate_cone(profile, preset, config.mute)
    folder = study.folder / SURVEY
    records, _ = with_records(study, folder, preset, profile, settings, config)
    # The surface waves' cone as a mute, for G1 to find the arrivals faster than it.
    fast = mute_candidates(cone, config.mute, profile)["cone"] if cone is not None else None
    _, usable, g1 = settled(folder, profile, preset, records, config, fast, study)
    xs = [receiver.x for receiver in profile.receivers]
    first, last = min(xs), max(xs)
    bands = [band for band in usable.values() if band is not None]
    durations = [record.duration_s for record in profile.records]
    segment = (
        float(preset.model_dump(mode="json")["slicing"]["segment_duration"]) if passive else None
    )
    found = Survey(
        profile=profile.name,
        receivers=len(profile.receivers),
        spacing_m=receiver_spacing(profile),
        line_m=round(last - first, 3),
        shots=tuple(
            _shot(
                record.path.name,
                record.source.x if record.source else None,
                first,
                last,
                record.trigger_s,
            )
            for record in profile.records
        ),
        sampling_hz=profile.sampling_rate_hz,
        nyquist_hz=profile.nyquist_hz,
        record_s=(min(durations), max(durations)),
        frequency_step_hz=round(1 / (segment or min(durations)), 4),
        shortest_wavelength_m=2 * receiver_spacing(profile),
        lengths=(SHORTEST_WINDOW, len(profile.receivers) // 2),
        reach_m=records.reach_m,
        cone_m_s=(cone.vmin, cone.median, cone.vmax) if cone is not None else None,
        cone_traces=cone.n_traces if cone is not None else None,
        usable_hz=(
            (
                statistics.median(low for low, _ in bands),
                statistics.median(high for _, high in bands),
            )
            if bands
            else None
        ),
        usable_spread_hz=(
            (max(low for low, _ in bands), min(high for _, high in bands)) if bands else None
        ),
        g1=g1,
        kind="passive" if passive else "active",
        modes=tuple(str(mode) for mode in MODES[profile.kind]),
        segment_s=segment,
    )
    write_atomic(saved, found.model_dump_json(indent=2))
    return found


def describe_survey(found: Survey) -> str:
    """The survey as the model reads it: a line of many shots or records summed up, the records
    G1 flags each said."""
    shots = shots_said(found.shots)
    low, high = found.lengths
    counts = ", ".join(
        f"{found.receivers - length + 1} at {length}"
        for length in sorted({low, (low + high) // 2, high})
    )
    limits = (
        f"Limits: no aliasing above a wavelength of {found.shortest_wavelength_m:g} m (2·dx), the "
        f"alias velocity {found.shortest_wavelength_m:g} m times f; windows of {low} to {high} "
        f"receivers, N_r - N + 1 of them at step 1 ({counts})."
    )
    modes = (
        "Modes: "
        + ", ".join(f"{mode} ({MODE_SAID[mode]})" for mode in found.modes if mode in MODE_SAID)
        + "."
    )
    if found.kind == "passive":
        return "\n".join(
            [
                f"{found.profile}: {found.receivers} receivers {found.spacing_m:g} m apart, "
                f"{found.line_m:g} m long; {len(found.shots)} records of ambient noise "
                f"({_records(found.shots)}), {_range(found.record_s, ' s')} long, no shot.",
                f"Sampled at {found.sampling_hz:g} Hz (Nyquist {found.nyquist_hz:g} Hz); "
                f"segments of {found.segment_s or 0:g} s (the preset's), an image's frequency step "
                f"{found.frequency_step_hz:.3g} Hz: longer segments give a finer step and lower "
                "frequencies, fewer of them to stack.",
                limits,
                "Reach and cone: none, a passive line has no shot; each window correlates the "
                "noise of every record.",
                "Usable band (G1): not measured on noise; G2 measures each window's correlations.",
                modes,
                "G1 on the raw records:",
                *_g1(found.g1),
            ]
        )
    lines = [
        f"{found.profile}: {found.receivers} receivers {found.spacing_m:g} m apart, "
        f"{found.line_m:g} m long; {len(found.shots)} shots: {shots}.",
        f"Sampled at {found.sampling_hz:g} Hz (Nyquist {found.nyquist_hz:g} Hz); records "
        f"{_range(found.record_s, ' s')} long, an image's frequency step "
        f"{found.frequency_step_hz:.3g} Hz.",
        limits,
        "Reach: "
        + (
            f"the traces' median SNR falls under G1's limit {found.reach_m:g} m from the shot."
            if found.reach_m is not None
            else "the SNR stays above G1's limit along the whole line."
        ),
        "Surface waves' cone: "
        + (
            f"{_range((found.cone_m_s[0], found.cone_m_s[2]), ' m/s')}, median "
            f"{found.cone_m_s[1]:g} m/s, from {found.cone_traces} traces."
            if found.cone_m_s is not None
            else "too few traces to tell."
        ),
        "Usable band (G1): "
        + (
            f"{_range(found.usable_hz, ' Hz')}, the records' median; "
            f"{_range(found.usable_spread_hz, ' Hz')} shared by all."
            if found.usable_hz is not None and found.usable_spread_hz is not None
            else "none found."
        ),
        modes,
        "G1 on the raw records:",
        *_g1(found.g1),
    ]
    return "\n".join(lines)


def shots_said(shots: Sequence[Shot]) -> str:
    """The shots, each said; past MANY, summed up: how many on each side, the line's at which
    positions, and their files' triggers."""
    if len(shots) <= MANY:
        return "; ".join(_said(shot) for shot in shots)
    said: list[str] = []
    for side in ("before the line", "on the line", "after the line", "unknown"):
        placed = sorted(shot.x_m for shot in shots if shot.side == side and shot.x_m is not None)
        count = sum(1 for shot in shots if shot.side == side)
        if not count:
            continue
        if side == "unknown":
            said.append(f"{count} at unknown places")
            continue
        steps = {round(b - a, 3) for a, b in pairwise(placed)}
        every = f" every {next(iter(steps)):g} m" if len(steps) == 1 else ""
        where = (
            f"at {placed[0]:g} m"
            if len(placed) == 1
            else f"from {placed[0]:g} to {placed[-1]:g} m{every}"
        )
        said.append(f"{count} {side} ({where})")
    triggers = sorted({shot.trigger_s for shot in shots if shot.trigger_s})
    if len(triggers) == 1:
        said.append(f"their files' trigger {triggers[0]:g} s")
    elif triggers:
        said.append(f"their files' triggers {triggers[0]:g} to {triggers[-1]:g} s")
    return ", ".join(said)


def _g1(results: Mapping[str, GateResult]) -> list[str]:
    """G1's results: the line's, the flagged records each with its measures (the first MANY), the
    others summed up by verdict."""
    lines: list[str] = []
    plain: dict[str, int] = {}
    flagged = 0
    for unit, result in results.items():
        if unit != LINE and not result.flags:
            plain[result.verdict] = plain.get(result.verdict, 0) + 1
            continue
        if unit != LINE:
            flagged += 1
            if flagged > MANY:
                continue
        flags = ", ".join(flag.name for flag in result.flags) or "no flag"
        measures = ", ".join(
            f"{metric.name} {metric.value:.3g}{metric.unit}"
            for metric in result.metrics
            if metric.value is not None
        )
        where = "the line" if unit == LINE else unit
        lines.append(
            f"- {where}: {result.verdict}; {flags}" + (f"; {measures}" if measures else "")
        )
    if flagged > MANY:
        lines.append(f"- {flagged - MANY} records more flagged (observe or look gives them).")
    lines.extend(
        f"- {count} record{'s' if count != 1 else ''}: {verdict}, no flag"
        for verdict, count in plain.items()
    )
    return lines


def _shot(record: str, x: float | None, first: float, last: float, trigger_s: float | None) -> Shot:
    """A record's shot, against the line from `first` to `last` receiver."""
    if x is None:
        return Shot(record=record, x_m=None, side="unknown", offset_m=None, trigger_s=trigger_s)
    if x < first:
        side, offset = "before the line", first - x
    elif x > last:
        side, offset = "after the line", x - last
    else:
        side, offset = "on the line", 0.0
    return Shot(record=record, x_m=x, side=side, offset_m=round(offset, 3), trigger_s=trigger_s)


def _said(shot: Shot) -> str:
    if shot.x_m is None:
        return f"{shot.record} at an unknown place"
    where = f"{shot.offset_m:g} m {shot.side}" if shot.side != "on the line" else "on the line"
    trigger = f", its file's trigger {shot.trigger_s:g} s" if shot.trigger_s else ""
    return f"{shot.record} at {shot.x_m:g} m ({where}{trigger})"


def _records(shots: Sequence[Shot]) -> str:
    """A passive line's records by name: each, or the first and the last past MANY."""
    names = [shot.record for shot in shots]
    return ", ".join(names) if len(names) <= MANY else f"{names[0]} to {names[-1]}"


def _range(values: tuple[float, float] | None, unit: str = "") -> str:
    if values is None:
        return "none"
    low, high = values
    return f"{low:.3g}{unit}" if low == high else f"{low:.3g} to {high:.3g}{unit}"
