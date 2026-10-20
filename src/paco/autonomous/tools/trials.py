"""Trials (docs/autonomous_plan.md §4.2 to §4.5, §5.3, §6 try_settings): variants of the line's settings,
each tried on a set of the line's windows, judged by G1 to G3 and scored, changing no run.

A variant's windows are the line's own: one receiver apart, each stacking the shots the line's
rules give it (masw.distance_min and distance_max, and the near-field distance), or on a passive
line correlating the noise of every record, so that their images are those the line would make,
and the images' cache gives them back when the line is committed with the same settings. The line
is processed in a mode: its profile's own (active, passive), or passive-active on an active line,
its shots correlated. Every record is preprocessed, so that G1 leaves out the traces
and records the line would; a variant that changes no preprocessing reuses its records. G2 and G3
judge each window as sensors: nothing they suggest is applied. A trial made before, the same
settings on the same windows, is not made again: its result comes back.

The study's folder holds it all (<output>/<profile>/studies/<id>/):
- records/<key>/: the records preprocessed one way, for every variant that preprocesses them so;
- g1/<key>/: G1 on the records preprocessed one way, made once for the images' band, the reach,
  the cone and the QC configuration it was made with, and given to every trial and commit alike
  (settled.json, each record's spectra beside);
- trials/<key>/: a variant on its windows: their images and automatic picks (xmid_*/), G1's
  log, result.json;
- experiments/: each call, E<n>.json, and index.json, the calls by their content."""

import contextvars
import hashlib
import json
import re
import shutil
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.base.dispersion_curve import DispersionCurvesImage
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.dataio.selection_plotting import load_selection
from sigpipe.dataio.signal_plotting import SPECTRA_STEM
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.pipelines import PREPROCESSED
from sigpipe.masw.presets import (
    ActivePreset,
    PassiveActivePreset,
    PassivePreset,
    make_preset,
    resolve_preset,
)
from sigpipe.masw.profiles import MODES, Profile, load_profile
from sigpipe.masw.quality.measures import measure_signal, selection_measures
from sigpipe.masw.runs import (
    RecordOutcome,
    RunError,
    WindowOutcome,
    caching,
    load_image,
    xmid_of,
)
from sigpipe.masw.runs.processing import (
    RECORDS_FOLDER,
    new_run_folder,
    preprocess_records,
    process_windows,
)
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import Exclusions, MASWWindow, build_windows

from paco import stopping
from paco.autonomous.pins import paco_code
from paco.autonomous.sensors.scorecard import Against, Scorecard, against, dominated, scorecard
from paco.autonomous.tools.given import check_line, given_of, with_line
from paco.qc.coherence import TrialJudge, near_field_windows, trial_curve, trial_indices
from paco.qc.config import QCConfig
from paco.qc.g2_image import judge_image, more_data
from paco.qc.g4_profile import LINE
from paco.qc.judging import CORRELATION_MODES, image_band, shared_band, stream_of
from paco.qc.line import line_reach, settle_records
from paco.qc.log import append_attempt, latest, read_attempts
from paco.qc.loops import deep_merge
from paco.qc.models import Attempt, GateResult
from paco.qc.muting import estimate_cone, mutable, mute_candidates
from paco.runs import image_cache
from paco.settings import Settings

STUDIES = "studies"  # in the profile's output folder
RECORDS = "records"
G1 = "g1"
SETTLED = "settled.json"  # in a folder of g1/
TRIALS = "trials"
EXPERIMENTS = "experiments"
RESULT = "result.json"
INDEX = "index.json"
# The frozen curves (freeze): once written, the study is past its curves (P4).
CURVES = "curves.json"
# The survey's file (survey.SAVED): what it measured of the line, its reach among it.
SURVEY_FILE = "survey.json"
# The most variants a call tries, besides the current settings (§6), and its windows by default:
# images and picks are cheap, a sweep compares many settings on enough windows.
MOST_VARIANTS = 12
# Trials made at once, each with its share of the workers; one at a time makes the records and G1
# the study shares.
AT_ONCE = 3
_PREPARING = threading.Lock()
WINDOWS = "spread:15"
# What a variant may change besides the preset's stages: the automatic pick's parameters and the
# near-field distance, m (a window stacks no shot nearer to its nearest receiver, where it has a
# farther one: PACo's rule, §4.3).
PICKING = "picking"
NEAR_FIELD = "near_field_m"
MODE = "mode"
# The preset's parts a record's preprocessing reads (sigpipe's preprocessing pipeline).
# The stages a record's preprocessing reads (a shot's trigger and mute, the filter): a change of
# one preprocesses the records again.
PREPROCESSING = ("trigger", "muting", "filtering")
SPEC = "spread:n, failing, flag:<name>, xmids:[x, ...] or holdout:n, joined by +"
# A study's id, as a run's: when it began, and four hex digits.
# A line's preset, in any mode.
type Preset = ActivePreset | PassivePreset | PassiveActivePreset
STUDY_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
_PART = re.compile(r"(spread|holdout):(\d+)|(failing)|flag:(\w+)|xmids:(\[.*\])")


@dataclass(frozen=True)
class Study:
    """The agent's study of one profile: its folder, <output>/<profile>/studies/<id>/."""

    profile: str
    folder: Path

    @property
    def id(self) -> str:
        return self.folder.name


@dataclass(frozen=True)
class Spread:
    """`n` windows spread along the line, its ends included."""

    n: int


@dataclass(frozen=True)
class Failing:
    """The windows whose curve fails G3 on the committed line."""


@dataclass(frozen=True)
class Flagged:
    """The windows whose curve carries flag `name` on the committed line."""

    name: str


@dataclass(frozen=True)
class At:
    """The windows nearest these positions, m."""

    xmids: tuple[float, ...]


@dataclass(frozen=True)
class Holdout:
    """`n` windows spread along the line, none of them in the last trial set: a check apart."""

    n: int


type Part = Spread | Failing | Flagged | At | Holdout


@dataclass(frozen=True)
class LineState:
    """What a trial set reads of the committed line: each window's G3 verdict and flags, by
    xmid."""

    verdicts: Mapping[float, str] = field(default_factory=dict[float, str])
    flags: Mapping[float, frozenset[str]] = field(default_factory=dict[float, frozenset[str]])


@dataclass(frozen=True)
class Trial:
    """A variant made concrete: its settings (the current ones with its changes), its preset
    (one receiver between windows), the automatic pick's parameters and the near-field
    distance."""

    settings: dict[str, Any]
    preset: Preset
    picking: PickingParameters
    near_field_m: float | None


class Records(BaseModel):
    """A line's records preprocessed one way: each one's outcome, the line's reach, and what they
    cost, CPU seconds."""

    model_config = ConfigDict(frozen=True)

    outcomes: tuple[RecordOutcome, ...]
    reach_m: float | None
    seconds: float


class Settled(BaseModel):
    """G1 on a line's records, as PACo's settle_records leaves a run folder: each record's result
    and the notes it added to the record's attempt, its result on the line, the traces and records
    it leaves out of the windows, and each record's usable band."""

    model_config = ConfigDict(frozen=True)

    results: dict[str, GateResult]  # by record, and on the line
    notes: dict[str, tuple[str, ...]]
    exclusions: Exclusions
    usable: dict[str, tuple[float, float] | None]


class WindowResult(BaseModel):
    """A trial window: G2 on its image, G3 on its automatic pick; its error when it has no
    image."""

    model_config = ConfigDict(frozen=True)

    xmid: float
    cached: bool  # its image taken from the images' cache
    error: str | None = None
    g2: GateResult | None = None
    g3: GateResult | None = None


class TrialRun(BaseModel):
    """A variant tried on its windows: what G1, G2 and G3 said, and what it cost when made."""

    model_config = ConfigDict(frozen=True)

    key: str
    settings: dict[str, Any]
    length: int  # its windows' receivers
    length_m: float  # their length, from the first receiver to the last
    g1: dict[str, GateResult]  # by record, and on the line
    windows: tuple[WindowResult, ...]
    seconds: dict[str, float]  # CPU seconds: records (when made by it), images, picking
    made_at: datetime
    mode: str = "active"  # the line's processing mode


class Tried(BaseModel):
    """A variant in an experiment: what it changes on the current settings, its trial, its
    scorecard, and how it compares."""

    model_config = ConfigDict(frozen=True)

    label: str  # "current", then the variants' numbers
    changes: dict[str, Any]
    trial: str  # its trial's key, in the study's trials/
    taken: bool  # its trial was made before: taken from the study
    seconds: float  # what its trial cost this call: 0 when taken
    scorecard: Scorecard
    against: Against | None  # against the current settings, on the windows both judged
    dominated_by: str | None


class Experiment(BaseModel):
    """One call: the current settings and the variants tried on the same trial set."""

    model_config = ConfigDict(frozen=True)

    id: str  # E<n>, in its study
    profile: str
    windows: str  # the trial set, as asked
    xmids: tuple[float, ...]  # the windows the current settings were tried on
    current: dict[str, Any]
    tried: tuple[Tried, ...]  # the current settings first
    made_at: datetime
    repeated: bool = False  # the same call made before: its results again


def check_open(study: Study) -> None:
    """Refused once the study's curves are frozen: no processing or picking after the freeze
    (docs/autonomous_plan.md §8, rule 5)."""
    if (study.folder / CURVES).exists():
        raise RunError(
            "The curves are frozen: no processing or picking after the freeze. The inversion's "
            "turn (P4)."
        )


def new_study(profile: str, settings: Settings) -> Study:
    """A new study of `profile`, its folder made: its id the time it began and four hex digits,
    as a run's."""
    load_profile(profile, settings)  # refused when the profile is not there
    _, folder = new_run_folder(settings.output_dir / profile / STUDIES)
    return Study(profile, folder)


def study_of(profile: str, settings: Settings, study_id: str | None = None) -> Study:
    """Study `study_id` of `profile`; without one, the profile's newest study, else a new one.
    Found by its id, never by a path."""
    load_profile(profile, settings)  # the profile exists: its name is no path either
    folder = settings.output_dir / profile / STUDIES
    if study_id is not None:
        if not STUDY_ID.fullmatch(study_id) or not (folder / study_id).is_dir():
            raise ValueError(f"No study '{study_id}' of profile '{profile}'.")
        return Study(profile, folder / study_id)
    found = sorted(path for path in folder.glob("*") if STUDY_ID.fullmatch(path.name))
    return Study(profile, found[-1]) if found else new_study(profile, settings)


def parse_windows(spec: str) -> tuple[Part, ...]:
    """The trial set `spec` names: `SPEC`."""
    parts: list[Part] = []
    for text in (piece.strip() for piece in spec.split("+")):
        found = _PART.fullmatch(text)
        if found is None:
            raise ValueError(f"Unknown trial set '{text}': {SPEC}.")
        spread, count, failing, flag, xmids = found.groups()
        if failing:
            parts.append(Failing())
        elif flag:
            parts.append(Flagged(flag))
        elif xmids:
            try:
                positions = tuple(float(x) for x in json.loads(xmids))
            except ValueError, TypeError:
                raise ValueError(f"'{text}': xmids is a list of positions, m.") from None
            parts.append(At(positions))
        elif int(count) < 1:
            raise ValueError(f"'{text}': at least one window.")
        else:
            parts.append(Spread(int(count)) if spread == "spread" else Holdout(int(count)))
    return tuple(parts)


def resolve(
    parts: Sequence[Part], windows: Sequence[MASWWindow], line: LineState, last: Sequence[float]
) -> list[MASWWindow]:
    """The `windows` (a variant's, the line's order) the parts name, each once: a spread over
    them, and the nearest to each position the others give (the committed line's windows, the
    positions asked, the last trial set's), so that a set holds for every window length."""
    xmids = [window.xmid for window in windows]
    chosen: set[int] = set()
    for part in parts:
        match part:
            case Spread(n):
                chosen.update(trial_indices(len(windows), n))
            case Failing() | Flagged():
                if not line.verdicts:
                    raise ValueError(
                        "failing and flag:<name> read the committed line's windows: commit one "
                        "first."
                    )
                found = (
                    [x for x, verdict in line.verdicts.items() if verdict != "pass"]
                    if isinstance(part, Failing)
                    else [x for x, flags in line.flags.items() if part.name in flags]
                )
                chosen.update(_nearest(xmids, x) for x in found)
            case At(positions):
                chosen.update(_nearest(xmids, x) for x in positions)
            case Holdout(n):
                taken = {_nearest(xmids, x) for x in last}
                free = [index for index in range(len(windows)) if index not in taken]
                if free:
                    chosen.update(free[at] for at in trial_indices(len(free), n))
    if not chosen:
        raise ValueError("The trial set names no window.")
    return [windows[index] for index in sorted(chosen)]


def concrete(
    settings: Mapping[str, Any], profile: Profile, config: QCConfig, reach_m: float | None
) -> Trial:
    """The line's `settings` (its mode, a preset's stages as sigpipe overrides them, `picking`
    and `near_field_m`) made concrete for `profile`; refused, saying what to change, when they
    cannot be. A window stacks no shot past the line's reach (`reach_m`, the survey's) unless the
    settings give masw.distance_max, as PACo's workflow sets it."""
    stages = {
        key: value for key, value in settings.items() if key not in (PICKING, NEAR_FIELD, MODE)
    }
    mode = str(settings.get(MODE) or profile.kind)
    modes = [str(one) for one in MODES[profile.kind]]
    if mode not in modes:
        raise ValueError(f"{MODE}: {' or '.join(modes)} on this {profile.kind} line, not {mode}.")
    masw = stages.get("masw")
    if isinstance(masw, Mapping) and cast(Mapping[str, Any], masw).get("step", 1) != 1:
        raise ValueError(
            "masw.step: the windows are one receiver apart, the line's and every trial's: change "
            "their length instead."
        )
    near = settings.get(NEAR_FIELD)
    if near is not None and (not isinstance(near, int | float) or near < 0):
        raise ValueError(f"{NEAR_FIELD}: a distance in metres, 0 or more, or null for none.")
    if near is not None and mode != "active":
        raise ValueError(
            f"{NEAR_FIELD}: the active mode's, whose windows stack shots; none in the {mode} mode."
        )
    picking = settings.get(PICKING) or {}
    if not isinstance(picking, Mapping):
        raise ValueError(f"{PICKING}: the automatic pick's parameters, by name.")
    made = dict(settings)
    if (muting := _mute_completed(stages.get("muting"), profile, config)) is not None:
        stages["muting"] = made["muting"] = muting
    windows: dict[str, Any] = {"step": 1}
    given = cast(Mapping[str, Any], masw) if isinstance(masw, Mapping) else {}
    if reach_m is not None and mode != "passive" and "distance_max" not in given:
        windows["distance_max"] = reach_m
    preset = resolve_preset(make_preset(mode, deep_merge(stages, {"masw": windows})), profile)
    parameters = PickingParameters.model_validate(
        {**config.picking.model_dump(), **cast(Mapping[str, Any], picking)}
    )
    return Trial(made, preset, parameters, float(near) if near is not None else None)


def survey_reach(study: Study) -> float | None:
    """The line's reach the study's survey measured, m: None before the survey, or where it found
    none."""
    path = study.folder / SURVEY_FILE
    if not path.exists():
        return None
    reach = json.loads(path.read_text()).get("reach_m")
    return float(reach) if isinstance(reach, int | float) else None


def _mute_completed(muting: object, profile: Profile, config: QCConfig) -> dict[str, Any] | None:
    """A mute giving no width or no taper, completed as PACo's mute trial makes its own: the wave
    train's tail kept after the slowest arrival and the rules' taper; the values given stay.
    None for any other muting."""
    if not isinstance(muting, Mapping):
        return None
    values = dict(cast(Mapping[str, Any], muting))
    if values.get("method") != "mute":
        return None
    values.setdefault("width", config.mute.width_s)
    values.setdefault("taper", max(0, round(config.mute.taper_s * profile.sampling_rate_hz)))
    return values


def line_windows(
    profile: Profile, trial: Trial, excluded: Collection[str] = ()
) -> list[MASWWindow]:
    """The line's windows with `trial`'s settings: one receiver apart, each stacking the shots
    masw.distance_min and distance_max give it, but those nearer than the near-field distance to
    its nearest receiver where it has a farther one whose record G1 keeps (`excluded`: those it
    leaves out; the active mode's); on a passive line, every record."""
    windows = build_windows(profile, trial.preset.masw)
    if trial.near_field_m is not None and trial.preset.mode == "active":
        windows, _ = near_field_windows(windows, trial.near_field_m, excluded)
    return windows


def preprocessed(
    folder: Path, preset: Preset, profile: Profile, settings: Settings, config: QCConfig
) -> tuple[Records, bool]:
    """Every record of `profile` preprocessed with `preset` into `folder` (records/<record>), and
    the line's reach: made the first time, reused after (records.json). Whether made now."""
    saved = folder / "records.json"
    if saved.exists():
        return Records.model_validate_json(saved.read_text()), False
    shutil.rmtree(folder, ignore_errors=True)  # what a stopped call left half made
    folder.mkdir(parents=True)
    outcomes = preprocess_records(
        preset, profile, folder, settings.workers, stop=stopping.current()
    )
    started = time.process_time()
    reach = line_reach(folder, profile, outcomes, config, preset)
    seconds = sum(outcome.duration_s or 0.0 for outcome in outcomes)
    records = Records(
        outcomes=outcomes,
        reach_m=reach,
        seconds=round(seconds + time.process_time() - started, 3),
    )
    write_atomic(saved, records.model_dump_json(indent=2))
    return records, True


def with_records(
    study: Study,
    folder: Path,
    preset: Preset,
    profile: Profile,
    settings: Settings,
    config: QCConfig,
) -> tuple[Records, bool]:
    """The study's records preprocessed as `preset` says, made the first time (preprocessed),
    given to run folder `folder` as its records/; whether made now."""
    dumped = preset.model_dump(mode="json")
    store = study.folder / RECORDS / _key(_preprocessing(dumped))
    records, made = preprocessed(store, preset, profile, settings, config)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / RECORDS_FOLDER).symlink_to(store / RECORDS_FOLDER, target_is_directory=True)
    return records, made


def fast_cone(profile: Profile, preset: Preset, config: QCConfig) -> Mapping[str, Any] | None:
    """The surface waves' cone, for G1 to check a record left unmuted for faster arrivals, as
    PACo's run_processing gives it."""
    if not mutable(preset):
        return None
    cone = estimate_cone(profile, preset, config.mute)
    return mute_candidates(cone, config.mute, profile)["cone"] if cone is not None else None


def settled(
    folder: Path,
    profile: Profile,
    preset: Preset,
    records: Records,
    config: QCConfig,
    cone: Mapping[str, Any] | None = None,
    study: Study | None = None,
) -> tuple[Exclusions, dict[str, tuple[float, float] | None], dict[str, GateResult]]:
    """G1 on `records`, from run folder `folder`, whose records/ holds them: the traces and
    records it leaves out of the windows, each record's usable band, and G1's result on each
    record and on the line, logged in `folder`'s QC log. With `cone` (the surface waves' mute),
    a record left unmuted is checked for arrivals faster than it; with `study`, G1 is made once
    and kept there (judge_records)."""
    now = datetime.now(UTC)
    for outcome in records.outcomes:
        append_attempt(
            folder,
            Attempt(
                unit=outcome.name,
                stage="preprocessing",
                attempt=1,
                parameters={},
                triggered_by="initial",
                started_at=now,
                finished_at=now,
                status=outcome.status,
                error=outcome.error,
            ),
        )
    exclusions, usable = judge_records(
        folder, profile, preset, records.outcomes, config, records.reach_m, cone, study
    )
    return exclusions, usable, _g1(read_attempts(folder), records.outcomes)


def judge_records(
    folder: Path,
    profile: Profile,
    preset: Preset,
    outcomes: Sequence[RecordOutcome],
    config: QCConfig,
    reach_m: float | None,
    cone: Mapping[str, Any] | None = None,
    study: Study | None = None,
) -> tuple[Exclusions, dict[str, tuple[float, float] | None]]:
    """PACo's settle_records on the records of run folder `folder`, each logged there: the traces
    and records G1 leaves out of the windows, and each record's usable band; its results logged.
    With `study`, G1 is made once for what it depends on (the records' preprocessing and the
    code, the images' band, the reach, the cone, the QC configuration) and kept in the study,
    then given back: logged as settle_records logs it, each record's spectra copied beside it
    where `folder` holds records of its own (a trial's are the study's)."""
    if study is None:
        _, exclusions, usable = settle_records(
            folder, profile, preset, tuple(outcomes), config, reach_m, cone
        )
        return exclusions, usable
    dumped = preset.model_dump(mode="json")
    entry = (
        study.folder
        / G1
        / _key(
            _preprocessing(dumped)
            | {
                "paco": paco_code(),
                "band": image_band(dumped),
                "reach_m": reach_m,
                "cone": cone,
                "qc": config.model_dump(mode="json"),
            }
        )
    )
    own = not (folder / RECORDS_FOLDER).is_symlink()
    if (entry / SETTLED).exists():
        kept = Settled.model_validate_json((entry / SETTLED).read_text())
        _relog(folder, kept)
        if own:
            _copy_spectra(entry, folder, outcomes)
        return kept.exclusions, kept.usable
    before = read_attempts(folder)
    _, exclusions, usable = settle_records(
        folder, profile, preset, tuple(outcomes), config, reach_m, cone
    )
    after = read_attempts(folder)
    notes: dict[str, tuple[str, ...]] = {}
    for outcome in outcomes:
        then = latest(before, outcome.name, "preprocessing")
        now = latest(after, outcome.name, "preprocessing")
        added = now.notes[len(then.notes) if then is not None else 0 :] if now is not None else ()
        if added:
            notes[outcome.name] = added
    shutil.rmtree(entry, ignore_errors=True)  # what a stopped call left half made
    _copy_spectra(folder, entry, outcomes)
    kept = Settled(results=_g1(after, outcomes), notes=notes, exclusions=exclusions, usable=usable)
    write_atomic(entry / SETTLED, kept.model_dump_json(indent=2))
    return exclusions, usable


def image(
    windows: list[MASWWindow],
    preset: Preset,
    records: Records,
    folder: Path,
    exclusions: Exclusions,
    settings: Settings,
) -> tuple[WindowOutcome, ...]:
    """`windows`' images made into `folder` from the records its records/ holds, without G1's
    `exclusions`: taken from the images' cache when made before alike, kept in it."""
    with caching.using(image_cache(settings)):
        return process_windows(
            preset,
            windows,
            records.outcomes,
            folder,
            settings.workers,
            exclusions=exclusions,
            stop=stopping.current(),
        )


def trial_windows(
    trial: Trial,
    profile: Profile,
    parts: Sequence[Part],
    line: LineState | None,
    last: Sequence[float],
) -> list[MASWWindow]:
    """The windows of `trial` the trial set `parts` names; refused when no window has a shot."""
    line_wins = line_windows(profile, trial)
    if not line_wins:
        raise ValueError(
            f"No window of {trial.preset.masw.length} receivers has a shot within "
            "masw.distance_min and distance_max: widen them."
        )
    return resolve(parts, line_wins, line or LineState(), last)


def tried(
    study: Study,
    trial: Trial,
    windows: list[MASWWindow],
    profile: Profile,
    settings: Settings,
    config: QCConfig,
) -> tuple[TrialRun, bool]:
    """`trial` on `windows`: its result, made now or taken from the study when made before (the
    same settings on the same windows, by the same code), and whether it was taken."""
    preset = trial.preset.model_dump(mode="json")
    key = _key(
        {
            "code": caching.code_version(),
            "profile": study.profile,
            "preset": preset,
            "picking": trial.picking.model_dump(mode="json"),
            "near_field_m": trial.near_field_m,
            "windows": [_xmid(window) for window in windows],
            "qc": config.model_dump(mode="json"),
        }
    )
    folder = study.folder / TRIALS / key
    if (folder / RESULT).exists():
        return TrialRun.model_validate_json((folder / RESULT).read_text()), True
    shutil.rmtree(folder, ignore_errors=True)
    # The study's records and G1 are shared by every trial that preprocesses alike: made once.
    with _PREPARING:
        records, made = with_records(study, folder, trial.preset, profile, settings, config)
        cone = fast_cone(profile, trial.preset, config)
        exclusions, usable, g1 = settled(
            folder, profile, trial.preset, records, config, cone, study
        )
    if exclusions.records:
        # The near-field rule on the records G1 keeps: a window whose farther shots it left out
        # keeps its near ones.
        built = {window.xmid: window for window in line_windows(profile, trial, exclusions.records)}
        windows = [built[window.xmid] for window in windows]
    outcomes = image(windows, trial.preset, records, folder, exclusions, settings)
    started = time.process_time()
    judge = TrialJudge(config.coherence, config.curve, trial.picking, config.image)
    results = tuple(_judged(folder, outcome, trial, judge, usable, config) for outcome in outcomes)
    xs = [receiver.x for receiver in windows[0].acquisitions[0].receivers]
    run = TrialRun(
        key=key,
        mode=str(trial.preset.mode),
        settings=trial.settings,
        length=trial.preset.masw.length,
        length_m=round(max(xs) - min(xs), 3),
        g1=g1,
        windows=results,
        seconds={
            "records": records.seconds if made else 0.0,
            "images": round(sum(o.duration_s or 0.0 for o in outcomes if not o.cached), 3),
            "picking": round(time.process_time() - started, 3),
        },
        made_at=datetime.now(UTC),
    )
    write_atomic(folder / RESULT, run.model_dump_json(indent=2))
    return run, False


def tried_all(
    study: Study,
    trials: Sequence[tuple[Trial, list[MASWWindow]]],
    profile: Profile,
    settings: Settings,
    config: QCConfig,
) -> list[tuple[TrialRun, bool]]:
    """Each trial on its windows (tried): the first alone, on every worker, its images then in the
    cache for the others that share them; the others AT_ONCE at a time, each with its share of the
    workers, since a variant's pool starts slowly and its few windows would leave the machine
    idle."""
    if not trials:
        return []
    first = tried(study, *trials[0], profile, settings, config)
    rest = trials[1:]
    at_once = min(AT_ONCE, len(rest), settings.workers)
    if at_once < 2:
        return [first, *(tried(study, *one, profile, settings, config) for one in rest)]
    each = settings.model_copy(update={"workers": max(1, settings.workers // at_once)})
    with ThreadPoolExecutor(at_once) as pool:
        made = [
            pool.submit(
                contextvars.copy_context().run, tried, study, trial, windows, profile, each, config
            )
            for trial, windows in rest
        ]
        return [first, *(one.result() for one in made)]


def try_settings(
    study: Study,
    variants: Sequence[Mapping[str, Any]],
    windows: str = WINDOWS,
    *,
    current: Mapping[str, Any] | None = None,
    line: LineState | None = None,
    settings: Settings,
    config: QCConfig,
) -> Experiment:
    """`variants`, each a change of the line's `current` settings (none: the preset's), tried
    with the current settings themselves on the trial set `windows` names, each judged by G1 to
    G3 and scored; the windows each gains and loses against the current settings, and the
    variants another one dominates. The settings the user gave stay in every variant, and a
    variant changing one is refused. The same call made before returns its experiment again."""
    check_open(study)
    if not variants:
        raise ValueError("Give a variant at least: a change of the current settings.")
    if len(variants) > MOST_VARIANTS:
        raise ValueError(f"At most {MOST_VARIANTS} variants a call.")
    parts = parse_windows(windows)
    profile = load_profile(study.profile, settings)
    # The user's settings in every variant; a variant changing one refused (§8, rule 1).
    user = given_of(study)
    base = with_line(current or {}, user)
    changes = [dict(change) for change in variants]
    for change in changes:
        check_line(change, user)
    reach = survey_reach(study)
    trials = [
        concrete(deep_merge(base, change), profile, config, reach) for change in [{}, *changes]
    ]
    last = last_xmids(study)
    chosen = [(trial, trial_windows(trial, profile, parts, line, last)) for trial in trials]
    stopping.check()
    runs = tried_all(study, chosen, profile, settings, config)
    call = _key({"windows": windows, "trials": [run.key for run, _ in runs]})
    index = _index(study)
    if call in index["calls"]:
        # Made before: its results again, every trial taken from the study, nothing spent.
        made = Experiment.model_validate_json(
            (study.folder / EXPERIMENTS / f"{index['calls'][call]}.json").read_text()
        )
        taken = tuple(one.model_copy(update={"taken": True, "seconds": 0.0}) for one in made.tried)
        return made.model_copy(update={"repeated": True, "tried": taken})
    experiment = _experiment(f"E{len(index['calls']) + 1}", study, windows, base, changes, runs)
    write_atomic(
        study.folder / EXPERIMENTS / f"{experiment.id}.json", experiment.model_dump_json(indent=2)
    )
    index["calls"][call] = experiment.id
    write_atomic(study.folder / EXPERIMENTS / INDEX, json.dumps(index, indent=2))
    return experiment


def describe(experiment: Experiment) -> str:
    """`experiment` as the model reads it: one line a variant, the current settings first."""
    variants = len(experiment.tried) - 1
    lines = [
        f"{experiment.id}: {variants} variant{'s' if variants > 1 else ''} on "
        f"{experiment.windows}, against the current settings"
        + (" (made before)" if experiment.repeated else "")
        + ":"
    ]
    for one in experiment.tried:
        card = one.scorecard
        head = f"{one.label} {json.dumps(one.changes)}" if one.changes else one.label
        passing = f"{len(card.covered)} of {len(card.windows)} windows pass G3"
        if one.against is not None:
            passing += f" (+{len(one.against.gained)}, -{len(one.against.lost)})"
        marks = [f"dominated by {one.dominated_by}"] if one.dominated_by else []
        if one.against is not None and one.against.shortened:
            marks.append("gains by shortening its curves (rule 6)")
        cost = "made before" if one.taken else f"{one.seconds:.1f} s"
        lines.append(
            f"- {head}: {passing}; {figures(card)}; {cost}" + "".join(f"; {mark}" for mark in marks)
        )
    return "\n".join(lines)


def figures(card: Scorecard) -> str:
    """A scorecard's figures, as an experiment says them: the images (their score, its weakest
    windows' and the measures known of it: an active image has no lateral coherence), then
    depth, span, uncertainty, the near field (a passive line's windows have no shot: none),
    beyond reach and aliased."""
    measured = [
        said
        for said, value in (
            (f"coherent {_share(card.coherent)}", card.coherent),
            (f"lateral {_share(card.lateral)}", card.lateral),
            (f"on the ridge {_share(card.on_ridge)}", card.on_ridge),
            (f"background {_value(card.background)}", card.background),
            (f"stripes {_value(card.stripes)}", card.stripes),
        )
        if value is not None
    ]
    weakest = f"P10 {card.image_p10:.2f}; " if card.image_p10 is not None else ""
    images = (
        f"image {card.image:.2f} ({weakest}{', '.join(measured)}); "
        if card.image is not None
        else ""
    )
    if not card.covered:
        return f"{images}no curve to score"
    near = f"near field {_share(card.near_field)}, " if card.near_field is not None else ""
    return (
        f"{images}depth {_value(card.depth_m, ' m')} (P10 {_value(card.depth_p10_m, ' m')}), "
        f"span {_value(card.span)}, uncertainty {_share(card.precision)}; {near}beyond reach "
        f"{_share(card.beyond)}, aliased {_share(card.aliased)}"
    )


def card_of(run: TrialRun) -> Scorecard:
    """`run`'s scorecard, as an experiment scores its variants."""
    return scorecard(
        {result.xmid: result.g3 for result in run.windows},
        run.length_m,
        cpu_s=round(sum(run.seconds.values()), 3),
        shots=run.mode != "passive",
        g2={result.xmid: result.g2 for result in run.windows},
    )


def _experiment(
    name: str,
    study: Study,
    windows: str,
    base: dict[str, Any],
    changes: list[dict[str, Any]],
    runs: list[tuple[TrialRun, bool]],
) -> Experiment:
    """The experiment of `runs`, the current settings' first, then the variants' in order."""
    labels = ["current", *(str(number) for number in range(1, len(changes) + 1))]
    cards = {label: card_of(run) for label, (run, _) in zip(labels, runs, strict=True)}
    beaten = dominated(cards)
    return Experiment(
        id=name,
        profile=study.profile,
        windows=windows,
        xmids=tuple(result.xmid for result in runs[0][0].windows),
        current=base,
        tried=tuple(
            Tried(
                label=label,
                changes=change,
                trial=run.key,
                taken=taken,
                seconds=0.0 if taken else round(sum(run.seconds.values()), 3),
                scorecard=cards[label],
                against=against(cards[label], cards["current"]) if label != "current" else None,
                dominated_by=beaten.get(label),
            )
            for label, change, (run, taken) in zip(labels, [{}, *changes], runs, strict=True)
        ),
        made_at=datetime.now(UTC),
    )


def _judged(
    folder: Path,
    outcome: WindowOutcome,
    trial: Trial,
    judge: TrialJudge,
    usable: dict[str, tuple[float, float] | None],
    config: QCConfig,
) -> WindowResult:
    """G2 on a trial window's image, with its records' usable band, and on a passive or
    passive-active line its correlations and FK selection; G3 on its automatic pick, kept beside
    the image for look to draw."""
    xmid = xmid_of(outcome.folder)
    if outcome.status != "succeeded":
        return WindowResult(xmid=xmid, cached=False, error=outcome.error)
    window = folder / outcome.folder
    mode = str(trial.preset.mode)
    values = trial.preset.model_dump(mode="json")
    muted = (values.get("muting") or {}).get("method", "none") != "none"
    g2 = judge_image(
        outcome.folder,
        load_image(window),
        config.image,
        shared_band(window, usable),
        mode=mode,
        muted=muted,
        **_correlations(window, mode, values, muted, config),
    )
    # A line that can be muted has its mute to try: a flag only a mute fixes is fixable in a
    # trial; a passive line's is not.
    g3, curve = trial_curve(window, judge, mutable="muting" in type(trial.preset).model_fields)
    if curve is not None:
        save_dispersion_curves(
            DispersionCurvesImage(dispersion_curves=(curve,)), path=window / CURVES_FILE
        )
    return WindowResult(xmid=xmid, cached=outcome.cached, g2=g2, g3=g3)


def _correlations(
    window: Path, mode: str, values: Mapping[str, Any], muted: bool, config: QCConfig
) -> dict[str, Any]:
    """G2's arguments on a passive or passive-active trial window (none for an active one), as
    PACo's correlations_of gives them on a run's, from the trial's settings (`values`): its
    stacked correlations measured as a record is, from the virtual source (not their SNR nor band
    when its records were `muted`), a passive window's FK selection, and the phase shift asked
    again with more of the data."""
    if mode not in CORRELATION_MODES:
        return {}
    path = window / PREPROCESSED
    correlations = (
        measure_signal(
            stream_of(path),
            config.signal,
            source="virtual",
            spectra=True,
            records_muted=muted,
            image_band=image_band(values),
        )
        if path.exists()
        else None
    )
    selection = load_selection(window)
    return {
        "correlations": correlations,
        "selection": selection_measures(selection, config.signal) if selection else (),
        "more": more_data(mode, values),
    }


def _index(study: Study) -> dict[str, dict[str, str]]:
    """The study's experiments, by the content of their calls."""
    path = study.folder / EXPERIMENTS / INDEX
    if not path.exists():
        (study.folder / EXPERIMENTS).mkdir(parents=True, exist_ok=True)
        return {"calls": {}}
    return json.loads(path.read_text())


def last_xmids(study: Study) -> tuple[float, ...]:
    """The windows of the study's last trial set, those its current settings were tried on."""
    calls = _index(study)["calls"]
    if not calls:
        return ()
    last = max(calls.values(), key=lambda name: int(name.removeprefix("E")))
    made = Experiment.model_validate_json((study.folder / EXPERIMENTS / f"{last}.json").read_text())
    return made.xmids


def _nearest(xmids: Sequence[float], position: float) -> int:
    return min(range(len(xmids)), key=lambda index: abs(xmids[index] - position))


def _xmid(window: MASWWindow) -> float:
    """The window's position as its folder names it (xmid_<x>, two decimals)."""
    return float(f"{window.xmid:.2f}")


def _preprocessing(preset: Mapping[str, Any]) -> dict[str, Any]:
    """What the records' preprocessing depends on: whether they are shots or noise (a
    passive-active line's are its shots, preprocessed as the active mode's), `preset`'s stages it
    reads (as dumped), and the code's version."""
    return {
        "records": "noise" if preset.get(MODE) == "passive" else "shots",
        **{part: preset.get(part) for part in PREPROCESSING},
        "code": caching.code_version(),
    }


def _g1(attempts: Sequence[Attempt], outcomes: Sequence[RecordOutcome]) -> dict[str, GateResult]:
    """G1's latest result on each record and on the line."""
    results: dict[str, GateResult] = {}
    for unit in (*(outcome.name for outcome in outcomes), LINE):
        attempt = latest(attempts, unit, "preprocessing")
        if attempt is not None and (result := attempt.results.get("G1")) is not None:
            results[unit] = result
    return results


def _relog(folder: Path, kept: Settled) -> None:
    """G1's `kept` results logged in run folder `folder`'s QC log as settle_records logs them:
    each record's latest preprocessing attempt appended again with its result, then with the
    notes G1 added; then G1's result on the line."""
    attempts = read_attempts(folder)
    for unit, result in kept.results.items():
        attempt = latest(attempts, unit, "preprocessing")
        if unit == LINE or attempt is None:
            continue
        judged = attempt.model_copy(update={"results": {**attempt.results, result.gate: result}})
        append_attempt(folder, judged, "verdict")
        if notes := kept.notes.get(unit):
            append_attempt(
                folder, judged.model_copy(update={"notes": (*judged.notes, *notes)}), "notes"
            )
    if (line := kept.results.get(LINE)) is not None:
        now = datetime.now(UTC)
        append_attempt(
            folder,
            Attempt(
                unit=LINE,
                stage="preprocessing",
                attempt=1,
                parameters={},
                triggered_by="initial",
                started_at=now,
                finished_at=now,
                status="succeeded",
                results={line.gate: line},
            ),
        )


def _copy_spectra(source: Path, target: Path, outcomes: Sequence[RecordOutcome]) -> None:
    """Each record's spectra (sigpipe's figure and data) copied from its folder in `source` to
    its folder in `target`."""
    for outcome in outcomes:
        for path in sorted((source / outcome.folder).glob(f"{SPECTRA_STEM}_*")):
            (target / outcome.folder).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / outcome.folder / path.name)


def _key(content: Mapping[str, Any]) -> str:
    """A short hash of `content`, as JSON."""
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


def _value(value: float | None, unit: str = "") -> str:
    return f"{value:.3g}{unit}" if value is not None else "none"


def _share(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "none"
