"""The line loop (docs/qc_workflow.md): the line's settings, one set for every record and every
window as a person sets them in PAC's pages, changed while a change gives more windows a curve G3
passes. The gates judge each record (G1) and each window (G2, and G3 on a trial pick of its
image); the changes they ask of the records or of the images are the line's to make
(`candidates`): grouped by what they change (the records' trigger delays as their median), each
tried, the one the most failing windows ask first, on windows spread among those asking it and
the others (`try_candidate`), and kept when it gives `min_gain` more of them a curve G3 passes
without losing the long wavelengths. paco.qc.line runs the loop and makes the whole line again
with each change kept; the picking of each window, after it, is the window's own."""

import json
import shutil
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from sigpipe.masw.presets import ActivePreset, PassivePreset, apply_overrides, resolve_preset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.runs import RecordOutcome, start_worker
from sigpipe.masw.runs.processing import RECORDS_FOLDER, preprocess_records
from sigpipe.masw.runs.stopping import finished
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import Exclusions, MASWWindow
from sigpipe.workers import one_thread_each

from paco import stopping
from paco.qc.coherence import TrialJudge, trial_indices, trial_pick, try_windows
from paco.qc.given import unlocked
from paco.qc.loops import deep_merge
from paco.qc.models import GateResult, Keep, Override

LINE_CHANGES_FOLDER = "line_trials"  # inside the run folder: a change's trial, while it runs
LINE_LOOP_FILE = "line_loop.json"
# The stages whose settings are the line's: the records' and the images'.
type LineStage = Literal["preprocessing", "phase_shift"]
LINE_STAGES: tuple[LineStage, ...] = ("preprocessing", "phase_shift")


class LineRules(BaseModel):
    """How the line loop decides: provisional, as the mute trial's rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_changes: int = Field(
        default=4,
        ge=0,
        description="Changes of the line's settings kept, at most: each makes the whole line "
        "again.",
    )
    min_gain: int = Field(
        default=2,
        ge=1,
        description="Trial windows a change must add to those whose curve passes G3: one is "
        "within what they vary by.",
    )
    trials: int = Field(
        default=15,
        ge=2,
        description="Windows a change is tried on: those asking it, spread, and as many of the "
        "others, spread along the line, when there are.",
    )
    max_wavelength_loss: float = Field(
        default=0.2,
        ge=0,
        lt=1,
        description="The share of the trial curves' median longest wavelength a change may "
        "lose: more, it cut what the line resolves deepest (over-muting the far offsets).",
    )


@dataclass(frozen=True)
class Candidate:
    """A change of the line's settings the gates asked: of the records (preprocessing) or of
    the images (phase_shift); the flag most of its windows raised ("<gate>:<flag>"); the windows
    asking it (for a record's flag, the windows using the record), in the line's order; and how
    many of them have a trial curve G3 did not pass."""

    stage: LineStage
    overrides: dict[str, Any]
    flag: str
    windows: tuple[str, ...]
    failing: int

    @property
    def key(self) -> str:
        return json.dumps([self.stage, self.overrides], sort_keys=True)


class LineTrial(BaseModel):
    """One change tried: on which trial windows, how many passed G3 with the line's settings
    and with the change, their curves' median longest wavelength, m, and whether it was kept."""

    model_config = ConfigDict(frozen=True)

    stage: LineStage
    overrides: dict[str, Any]
    flag: str
    asked_by: int  # windows asking it
    xmids: tuple[float, ...]
    before: int
    after: int
    longest_before: float | None = None
    longest_after: float | None = None
    kept: bool
    note: str


class LineLoop(BaseModel):
    """The line loop of a run (line_loop.json): each change tried, in order."""

    model_config = ConfigDict(frozen=True)

    trials: tuple[LineTrial, ...] = ()


@dataclass
class _Ask:
    stage: LineStage
    overrides: dict[str, Any]
    flags: Counter[str] = field(default_factory=Counter)
    windows: set[str] = field(default_factory=set)


def line_change(result: GateResult) -> list[tuple[LineStage, dict[str, Any], str]]:
    """The changes of the line's settings `result`'s flags ask: of the records or the images,
    each with its flag ("<gate>:<flag>"); never the windows themselves (masw)."""
    asked: list[tuple[LineStage, dict[str, Any], str]] = []
    for flag in result.flags:
        action = flag.action
        if (
            isinstance(action, Override)
            and action.stage in LINE_STAGES
            and action.overrides
            and "masw" not in action.overrides
        ):
            asked.append((action.stage, dict(action.overrides), f"{result.gate}:{flag.name}"))
    return asked


def candidates(
    records: Mapping[str, GateResult],
    images: Mapping[str, GateResult],
    picks: Mapping[str, GateResult],
    uses: Mapping[str, frozenset[str]],
    current: Mapping[str, Any],
    given: Mapping[str, Any],
    tried: frozenset[str],
) -> list[Candidate]:
    """The changes of the line's settings the gates ask (G1's of each record, by the windows
    using it in `uses`; G2's and the trial picks' G3 of each window), those a window whose trial
    curve failed asks, the one the most failing windows ask first: none that would leave the
    line's settings (`current`, the preset's values) as they are, touch a setting the user gave
    (`given`), or was `tried`. The records' trigger delays go as one change, their median."""
    failing = {unit for unit in uses if picks.get(unit) is None or picks[unit].verdict != "pass"}
    users: dict[str, set[str]] = {}
    for unit, names in uses.items():
        for name in names:
            users.setdefault(name, set()).add(unit)
    asks: dict[str, _Ask] = {}

    def add(stage: LineStage, overrides: dict[str, Any], flag: str, units: set[str]) -> None:
        ask = asks.setdefault(
            json.dumps([stage, overrides], sort_keys=True), _Ask(stage, overrides)
        )
        ask.flags[flag] += len(units)
        ask.windows |= units

    for results in (images, picks):
        for unit, result in results.items():
            for stage, overrides, flag in line_change(result):
                add(stage, overrides, flag, {unit})
    delays: list[float] = []
    delayed: set[str] = set()
    delay_flags: Counter[str] = Counter()
    for name, result in records.items():
        units = users.get(name, set())
        for stage, overrides, flag in line_change(result):
            t0 = _delay(overrides)
            if t0 is not None:
                delays.append(t0)
                delayed |= units
                delay_flags[flag] += 1
            elif units:
                add(stage, overrides, flag, units)
    if delays and delayed:
        median = round(statistics.median(delays), 4)
        ask = _Ask("preprocessing", {"trigger": {"t0": median}}, delay_flags, delayed)
        asks[json.dumps(["preprocessing", ask.overrides], sort_keys=True)] = ask
    found: list[Candidate] = []
    order = list(uses)
    for key, ask in asks.items():
        failed = len(ask.windows & failing)
        if not failed or key in tried:
            continue
        if deep_merge(current, ask.overrides) == dict(current):
            continue  # the line's settings already
        if unlocked(ask.overrides, given)[1]:
            continue  # a setting the user gave: theirs (U2)
        found.append(
            Candidate(
                stage=ask.stage,
                overrides=ask.overrides,
                flag=ask.flags.most_common(1)[0][0],
                windows=tuple(unit for unit in order if unit in ask.windows),
                failing=failed,
            )
        )
    return sorted(found, key=lambda one: (-one.failing, -len(one.windows), one.key))


def _delay(overrides: Mapping[str, Any]) -> float | None:
    """The trigger delay a record's change asks, alone in it; None for another change."""
    trigger = overrides.get("trigger")
    if set(overrides) != {"trigger"} or not isinstance(trigger, Mapping):
        return None
    t0 = cast(Mapping[str, Any], trigger).get("t0")
    return float(t0) if isinstance(t0, int | float) else None


def line_picks(
    run_folder: Path,
    units: Sequence[str],
    judge: TrialJudge,
    mutable: bool,
    workers: int,
) -> dict[str, GateResult]:
    """G3 on a trial pick of each window's image (coherence.trial_pick: the curves kept in
    memory, the run's picks untouched), in up to `workers` processes."""
    if workers <= 1 or len(units) <= 1:
        return {unit: trial_pick(run_folder / unit, judge, mutable) for unit in units}
    one_thread_each()  # the workers are the cores the picks take
    results: dict[str, GateResult] = {}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(units)), initializer=start_worker, initargs=(run_folder,)
    ) as executor:
        futures = {
            executor.submit(trial_pick, run_folder / unit, judge, mutable): unit for unit in units
        }
        for future in finished(executor, futures, stopping.current(), kill=False):
            results[futures[future]] = future.result()
    return {unit: results[unit] for unit in units if unit in results}


def try_candidate(
    candidate: Candidate,
    preset: ActivePreset | PassivePreset,
    profile: Profile,
    run_folder: Path,
    windows: Mapping[str, MASWWindow],
    picks: Mapping[str, GateResult],
    records: tuple[RecordOutcome, ...],
    exclusions: Exclusions,
    judge: TrialJudge,
    rules: LineRules,
    workers: int,
) -> LineTrial:
    """`candidate` tried on trial windows of the run's (`windows`, by folder, in the line's
    order): those asking it and the others, spread (`_sample`), made from records preprocessed
    with it (a change of the records) or from the run's (of the images), picked and judged by G3
    as the run's own were (`picks`); kept with `min_gain` more passing, the passed curves' median
    longest wavelength kept within `max_wavelength_loss`. Its trial files removed after."""
    chosen = _sample(list(windows), set(candidate.windows), rules.trials)
    tried_preset = resolve_preset(apply_overrides(preset, candidate.overrides), profile)
    folder = run_folder / LINE_CHANGES_FOLDER
    shutil.rmtree(folder, ignore_errors=True)
    try:
        if candidate.stage == "preprocessing":
            needed = sorted(
                {path.name for unit in chosen for path in windows[unit].selected_files}
                - set(exclusions.records)
            )
            tried_records = preprocess_records(
                tried_preset,
                profile,
                folder,
                workers,
                presets=dict.fromkeys(needed, tried_preset),
                stop=stopping.current(),
            )
            records_folder = folder / RECORDS_FOLDER
        else:
            tried_records = records
            records_folder = run_folder / RECORDS_FOLDER
        tried = try_windows(
            tried_preset,
            profile,
            [windows[unit] for unit in chosen],
            tried_records,
            records_folder,
            folder / "windows",
            judge,
            workers,
            exclusions,
            fix_grids=False,
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    before = sum(1 for unit in chosen if unit in picks and picks[unit].verdict == "pass")
    longest = [
        reach[1]
        for unit in chosen
        if unit in picks
        and picks[unit].verdict == "pass"
        and (reach := picks[unit].kept.wavelength_m) is not None
    ]
    longest_before = round(statistics.median(longest), 1) if longest else None
    longest_after = tried.wavelengths_m[1] if tried.wavelengths_m is not None else None
    lost = (
        longest_before is not None
        and longest_after is not None
        and longest_after < (1 - rules.max_wavelength_loss) * longest_before
    )
    kept = tried.passed - before >= rules.min_gain and not lost
    what = _said(candidate.overrides)
    asked = f"asked by {len(candidate.windows)} windows ({candidate.flag})"
    scores = f"{tried.passed} of {len(chosen)} trial windows passing G3 against {before}"
    if kept:
        # Worded as the line's other rules (the setting, then why): the parameters used read it.
        note = f"{what}: the line loop, {asked}: {scores} with the line's settings before it."
    else:
        why = (
            f"the curves' longest wavelengths {longest_after:g} m against {longest_before:g}"
            if lost and longest_after is not None and longest_before is not None
            else f"under {rules.min_gain} more"
        )
        note = f"Tried for the line, not kept: {what}, {asked}: {scores}, {why}."
    return LineTrial(
        stage=candidate.stage,
        overrides=candidate.overrides,
        flag=candidate.flag,
        asked_by=len(candidate.windows),
        xmids=tuple(windows[unit].xmid for unit in chosen),
        before=before,
        after=tried.passed,
        longest_before=longest_before,
        longest_after=longest_after,
        kept=kept,
        note=note,
    )


def _sample(order: Sequence[str], asking: set[str], trials: int) -> list[str]:
    """Up to `trials` windows, in the line's `order`: spread among those `asking`, and as many
    of the others, spread, when there are (the rest of the trials theirs when fewer ask)."""
    ask = [unit for unit in order if unit in asking]
    other = [unit for unit in order if unit not in asking]
    n_other = min(len(other), trials // 2)
    n_ask = min(len(ask), trials - n_other)
    n_other = min(len(other), trials - n_ask)
    chosen = {ask[index] for index in trial_indices(len(ask), n_ask)} if n_ask else set()
    if n_other:
        chosen |= {other[index] for index in trial_indices(len(other), n_other)}
    return [unit for unit in order if unit in chosen]


# The units of the settings a change words.
_UNITS = {"vmin": "m/s", "vmax": "m/s", "fmin": "Hz", "fmax": "Hz", "t0": "s", "width": "s"}


def _said(overrides: Mapping[str, Any]) -> str:
    """A change in words, as the line's rules word their settings: "muting mute 80 to 1500 m/s,
    width 0.05 s", "dispersion vmax 1500 m/s", "trigger t0 0.0029 s"."""
    parts: list[str] = []
    for stage, values in overrides.items():
        if not isinstance(values, Mapping):
            parts.append(f"{stage} {values}")
            continue
        given = dict(cast(Mapping[str, Any], values))
        if stage == "muting" and given.get("method") == "mute":
            del given["method"]
            low, high = given.pop("vmin", None), given.pop("vmax", None)
            head = f"muting mute {low:g} to {high:g} m/s" if low and high else "muting mute"
            rest = [f"{key} {_valued(key, value)}" for key, value in given.items()]
            parts.append(", ".join([head, *rest]))
            continue
        shown = ", ".join(f"{key} {_valued(key, value)}" for key, value in given.items())
        parts.append(f"{stage} {shown}")
    return "; ".join(parts)


def _valued(key: str, value: object) -> str:
    number = f"{value:g}" if isinstance(value, float) else str(value)
    return f"{number} {_UNITS[key]}" if key in _UNITS else number


def kept_on_line(result: GateResult) -> GateResult:
    """`result`, its unit's line done: the changes of the line's settings its flags ask, which
    the line loop did not keep, turned into kept notes (the line's settings stay the same for
    every unit); its verdict what its other flags make it, a pass when none acts."""
    asked = {flag for _, _, flag in line_change(result)}
    flags = tuple(
        flag.model_copy(
            update={"action": Keep(note="a change of the line's settings, which stay as they are")}
        )
        if f"{result.gate}:{flag.name}" in asked
        and isinstance(flag.action, Override)
        and flag.action.stage in LINE_STAGES
        else flag
        for flag in result.flags
    )
    acted = [
        flag
        for flag in flags
        if not isinstance(flag.action, Keep)
        and not (isinstance(flag.action, Override) and flag.action.stage == "picking")
    ]
    verdict = "pass" if not acted and result.verdict == "retry" else result.verdict
    return result.model_copy(update={"verdict": verdict, "flags": flags})


def write_line_loop(run_folder: Path, trials: Sequence[LineTrial]) -> None:
    """The changes the line loop tried, in line_loop.json."""
    write_atomic(
        run_folder / LINE_LOOP_FILE, LineLoop(trials=tuple(trials)).model_dump_json(indent=2)
    )
