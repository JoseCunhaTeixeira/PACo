"""search: the full grid of the line's settings, scored in code, for the model to choose from and
check that its choice makes sense. Every setting of PAC's page for
the line's method takes values the physics sets (axes), and every combination of them is a
variant: the windows' length over the line's range; the farthest shot as shares of the survey's
reach (the surface waves fall into the noise past it), never nearer than the window's half
length; the near field as shares of the window (a shot nearer than half a wavelength bends the
low frequencies); PACo's mutes from the survey's cone; a passive line's segments as crossings of
the window by the slowest wave, PACo's FK thresholds and bands, the whitening and the
normalization. Each variant is judged on the same trial windows as try_settings judges it, by the
grid (paco.autonomous.tools.grid). No score reads an image's quality as the eye does: the scores
only drop the really bad (no image, or too few windows passing for a curve along the line); every
other variant is judged by eye, in sheets of SHEET, one a row, each at up to grid.PLACES places
along the line, no score on them, the KEEP cleanest of each sheet kept (judge, which the host calls
from short calls of their own); the kept are mixed into new sheets, round after round, until KEEP
are left, the finalists, whose sheet, the final, goes to the agent itself: it chooses the cleanest
(choose), the search's choice. Nothing is tried after it: every combination was seen.

A study's search: searches/S1.json, each variant with its settings, trial and scorecard; its
thumbnails, searches/S1_images/<label>.npz; the knockout, searches/S1_knockout.json; each sheet's
judgment, searches/S1_judged.jsonl (what the eye kept and left)."""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from importlib.resources import files
from itertools import pairwise, product
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sigpipe.masw.profiles import MODES, Profile, load_profile
from sigpipe.masw.quality.cone import Cone
from sigpipe.masw.runs.writing import write_atomic

from paco.autonomous.tools import grid, trials
from paco.autonomous.tools import survey as surveying
from paco.autonomous.tools.given import Given, check_line, given_of, with_line
from paco.qc.config import QCConfig
from paco.qc.loops import deep_merge
from paco.settings import Settings, get_settings

SEARCHES = "searches"  # in the study's folder
SEARCH = "S1"  # a study's one search
IMAGES = f"{SEARCH}_images"
KNOCKOUT = f"{SEARCH}_knockout.json"
JUDGED = f"{SEARCH}_judged.jsonl"
WINDOWS = "spread:12"
# The grid's values, every one of them in a file beside the code.
GRID_FILE = "grid.yaml"
# The pseudo-settings a value may give, resolved by each variant's window (changes_of): a passive
# line's segments, their seconds per metre of the window; the near field, its share of the window.
SEGMENTS = "segment_s_per_m"
NEAR = "near_field_window"
# The really bad, which no sheet shows: no image, or fewer than this share of the windows passing.
PASSING = 0.25
# The sheets: variants a sheet, sheets a call, variants kept of a sheet (of fewer, all): the
# finalists once no more are left.
SHEET = 12
A_CALL = 3
KEEP = 3
# The line's settings each setting of the grid sets, by stage and key ("*": the whole stage).
SETS: dict[str, tuple[tuple[str, str], ...]] = {
    "length": (("masw", "length"),),
    "distance_min": (("masw", "distance_min"),),
    "distance_max": (("masw", "distance_max"),),
    "near_field_m": (("near_field_m", "*"),),
    "trigger": (("trigger", "*"),),
    "muting": (("muting", "*"),),
    "filtering": (("filtering", "*"),),
    "image_stacking": (("image_stacking", "*"),),
    "dispersion": (("dispersion", "*"),),
    "segments": (("slicing", "segment_duration"), ("slicing", "segment_step")),
    "fk_threshold": (("selection", "threshold"),),
    "fk_band": (("selection", "vmin"), ("selection", "vmax")),
    "whitening": (("whitening", "*"),),
    "normalization": (("normalization", "*"),),
    "stacking": (("stacking", "*"),),
}
# How a window says its FK selection kept no segment (sigpipe's Selection, the grid's alike).
KEPT_NONE = "fk selection kept none of the"


class Parameter(BaseModel):
    """A parameter of the grid: what it is, its unit, and its value where the grid fixes it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    unit: str
    value: Any = None


class Candidate(BaseModel):
    """A mute's velocities: two of the survey's cone (slow, median, fast) or of the grid's group
    velocities (slowest, fastest), the first divided and the second multiplied by `widen`."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    of: tuple[str, str] = Field(alias="from")
    widen: float = Field(default=1.0, ge=1.0)


class Setting(BaseModel):
    """A setting of the grid: what it is, its unit, the values tried; a stage's parameters (each
    its description and unit, a mute's its value); a length's longest, as a share of the line's
    receivers; a mute's candidates, by name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    unit: str
    values: tuple[Any, ...]
    parameters: dict[str, Parameter] = Field(default_factory=dict[str, Parameter])
    longest: float | None = None
    candidates: dict[str, Candidate | None] = Field(default_factory=dict[str, Candidate | None])


class Velocities(BaseModel):
    """The surface waves' group velocities the grid uses wherever it needs them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    slowest: Parameter
    fastest: Parameter


class Grid(BaseModel):
    """The grid, as grid.yaml gives it: the group velocities, the phase shift every method shares,
    then each method's settings in PAC's page order."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    group_velocities: Velocities
    dispersion: Setting
    active: dict[str, Setting]
    passive_active: dict[str, Setting] = Field(alias="passive-active")
    passive: dict[str, Setting]

    def method(self, mode: str) -> dict[str, Setting]:
        """The settings of `mode`, in PAC's page order, the phase shift among them: before an
        active line's images are stacked, after the correlations are."""
        own = {"active": self.active, "passive-active": self.passive_active}.get(mode, self.passive)
        found: dict[str, Setting] = {}
        for name, setting in own.items():
            if name == "image_stacking":
                found["dispersion"] = self.dispersion
            found[name] = setting
        found.setdefault("dispersion", self.dispersion)
        return found

    @model_validator(mode="after")
    def _known(self) -> Grid:
        for mode, own in (
            ("active", self.active),
            ("passive-active", self.passive_active),
            ("passive", self.passive),
        ):
            if "dispersion" in own:
                raise ValueError(f"{mode}: the dispersion is every method's, given at the top.")
            if unknown := sorted(set(own) - set(SETS)):
                raise ValueError(
                    f"{mode}: no setting {', '.join(unknown)} in the grid; its settings: "
                    f"{', '.join(SETS)}."
                )
            muting = own.get("muting")
            if muting is not None and (lost := set(muting.values) - set(muting.candidates)):
                raise ValueError(f"{mode} muting: no candidate {', '.join(sorted(lost))}.")
        return self


@cache
def _grid_file() -> dict[str, Any]:
    """grid.yaml, as written."""
    return yaml.safe_load((files("paco.autonomous") / GRID_FILE).read_text())


def grid_of(changes: Mapping[str, Any] | None = None) -> Grid:
    """The grid: grid.yaml's values with `changes` of them over the file's (a study's, where the
    user's words call for it), the file itself never written."""
    return Grid.model_validate(deep_merge(_grid_file(), dict(changes or {})))


@dataclass(frozen=True)
class Value:
    """A setting's value: its name, its number when it has one, and the changes of the line's
    settings it makes."""

    name: str
    number: float | None
    changes: Mapping[str, Any]


@dataclass(frozen=True)
class Axis:
    """A setting the grid moves, its values in order, and the line's settings it sets, by stage
    and key ("*": the whole stage)."""

    name: str
    values: tuple[Value, ...]
    sets: tuple[tuple[str, str], ...]
    unit: str = ""


class Row(BaseModel):
    """A variant of the grid: its value by setting, its changes, its trial and its scorecard."""

    model_config = ConfigDict(frozen=True)

    label: str  # g<n>
    values: dict[str, str]
    changes: dict[str, Any]
    trial: str
    length: int
    card: dict[str, Any]


class Search(BaseModel):
    """A study's search: its trial set, its settings and their values, every variant scored, how
    many could not be and why the first could not, the places along the line its sheets show,
    how many variants no sheet shows (the really bad)."""

    model_config = ConfigDict(frozen=True)

    id: str
    windows: str
    axes: dict[str, list[str]]
    units: dict[str, str]
    changes: dict[str, Any]  # of grid.yaml's values, for this study
    rows: tuple[Row, ...]
    refused: int
    refusal: str | None
    places: tuple[float, ...] = ()
    dropped: int = 0
    seconds: float
    made_at: datetime


class Sheet(BaseModel):
    """A sheet of the knockout: its number in its round, its variants in reading order."""

    model_config = ConfigDict(frozen=True)

    number: int
    labels: tuple[str, ...]


class Knockout(BaseModel):
    """The model's look at every variant: the round, its sheets, those judged and what each kept;
    once KEEP at most are left, the finalists, then their sheet, the final, and the one the eye
    chose of them: the search's choice. (A search of an older PACo tried its finalists side by
    side instead, as `experiment`.)"""

    model_config = ConfigDict(frozen=True)

    round: int
    sheets: tuple[Sheet, ...]
    kept: dict[int, tuple[str, ...]] = Field(default_factory=dict[int, tuple[str, ...]])
    finalists: tuple[str, ...] = ()
    best: str | None = None
    experiment: str | None = None

    @property
    def left(self) -> list[Sheet]:
        """This round's sheets still to judge."""
        return [sheet for sheet in self.sheets if sheet.number not in self.kept]

    @property
    def final(self) -> bool:
        """Whether this round is the final: the finalists' sheet, the one cleanest to choose."""
        return bool(self.finalists) and self.best is None

    @property
    def done(self) -> bool:
        """Whether the knockout has ended: its choice made (or its finalists tried, an older
        PACo's)."""
        return self.best is not None or self.experiment is not None


def searched(study: trials.Study) -> Search | None:
    """The study's search, when it has run."""
    path = study.folder / SEARCHES / f"{SEARCH}.json"
    return Search.model_validate_json(path.read_text()) if path.exists() else None


def knockout_of(study: trials.Study) -> Knockout | None:
    """The study's knockout, once its search has run."""
    path = study.folder / SEARCHES / KNOCKOUT
    return Knockout.model_validate_json(path.read_text()) if path.exists() else None


def unjudged(study: trials.Study) -> str | None:
    """Why the line's settings cannot be tried or committed yet: no search, or sheets of it the
    model has not judged; None once the search's choice is made (a search made before the sheets
    has none)."""
    if searched(study) is None:
        return (
            "Run search first: the full grid of the line's settings, every variant judged by eye, "
            "sheet by sheet, until the cleanest is chosen."
        )
    knockout = knockout_of(study)
    if knockout is None or knockout.done:
        return None
    if knockout.final:
        return (
            f"Choose first the cleanest of the search's {len(knockout.finalists)} finalists "
            "(choose), by eye on their dispersion images: search shows them again."
        )
    return (
        f"Every sheet is judged first: {len(knockout.left)} of round {knockout.round}'s "
        f"{len(knockout.sheets)} left; search again judges them."
    )


def choose(study: trials.Study, label: str, why: str) -> Knockout:
    """The search's final, the agent's own: `label`, the cleanest of the finalists by its eye,
    the search's choice, recorded with `why` as the sheets' judgments are."""
    knockout = knockout_of(study)
    if knockout is None or not knockout.final:
        raise ValueError(
            f"The search chose {knockout.best} already: commit it."
            if knockout is not None and knockout.best is not None
            else "No final to choose from yet: search judges the sheets until 3 are left."
        )
    if label not in knockout.finalists:
        raise ValueError(f"{label} is not a finalist: {', '.join(knockout.finalists)}.")
    return judge(study, {1: [label]}, why)


def chosen(study: trials.Study) -> str | None:
    """The variant the search chose by eye, once it has; None before, or for a search that
    chose none (no variant to judge, or an older PACo's)."""
    knockout = knockout_of(study)
    return knockout.best if knockout is not None else None


def to_keep(knockout: Knockout, sheet: Sheet) -> int:
    """How many variants of `sheet` the eye keeps: KEEP, all of a sheet of fewer; of the final,
    the one cleanest."""
    return 1 if knockout.final else min(KEEP, len(sheet.labels))


def judge(study: trials.Study, kept: Mapping[int, Sequence[str]], why: str) -> Knockout:
    """The sheets `kept` names judged: each sheet's kept variants (to_keep), recorded with `why`
    (JUDGED); a round judged, its kept mixed into the next round's sheets, or, KEEP at most left,
    the finalists and their sheet, the final; the final judged, its one kept the search's
    choice."""
    made, knockout = searched(study), knockout_of(study)
    if made is None or knockout is None:
        raise ValueError("No sheets to judge: run search first.")
    if knockout.done:
        raise ValueError(
            f"Every sheet is judged: the search chose {knockout.best}. Commit it."
            if knockout.best is not None
            else f"Every sheet is judged: the finalists {', '.join(knockout.finalists)}, "
            f"{knockout.experiment}."
        )
    if not kept:
        raise ValueError("kept: the variants you keep of each sheet, by its number.")
    left = {sheet.number: sheet for sheet in knockout.left}
    for number, labels in kept.items():
        sheet = left.get(number)
        if sheet is None:
            raise ValueError(
                f"Sheet {number} is not one left in round {knockout.round}: "
                f"{', '.join(str(one) for one in left)}."
            )
        most = to_keep(knockout, sheet)
        if len(labels) != most or len(set(labels)) != len(labels):
            raise ValueError(
                f"Sheet {number}: choose the one cleanest of its variants."
                if knockout.final
                else f"Sheet {number}: keep {most} of its variants, each once."
            )
        if lost := [label for label in labels if label not in sheet.labels]:
            raise ValueError(
                f"Sheet {number}: no variant {', '.join(lost)} on it; its: "
                f"{', '.join(sheet.labels)}."
            )
    at = datetime.now(UTC).isoformat()
    with (study.folder / SEARCHES / JUDGED).open("a") as file:
        for number, labels in kept.items():
            sheet = left[number]
            record = {
                "round": knockout.round,
                "sheet": number,
                "shown": list(sheet.labels),
                "kept": list(labels),
                "why": why,
                "at": at,
            }
            file.write(json.dumps(record) + "\n")
    judged = knockout.model_copy(
        update={"kept": {**knockout.kept, **{n: tuple(one) for n, one in kept.items()}}}
    )
    if judged.left:
        return _keep(study, judged)
    survivors = [label for sheet in judged.sheets for label in judged.kept[sheet.number]]
    if judged.final:
        return _keep(study, judged.model_copy(update={"best": survivors[0]}))
    return _keep(study, next_round(judged.round + 1, survivors))


def next_round(number: int, survivors: Sequence[str]) -> Knockout:
    """Round `number` of `survivors`: KEEP at most, they are the finalists, their sheet the final
    (one alone, the choice already); more, mixed into sheets of SHEET at most, dealt in turn, so
    that each sheet holds the kept of different sheets before."""
    if len(survivors) <= KEEP:
        finalists = tuple(survivors)
        sheets = (Sheet(number=1, labels=finalists),)
        if len(finalists) == 1:
            return Knockout(round=number, sheets=sheets, kept={1: finalists}, finalists=finalists,
                            best=finalists[0])  # fmt: skip
        return Knockout(round=number, sheets=sheets, finalists=finalists)
    count = -(-len(survivors) // SHEET)
    dealt: list[list[str]] = [[] for _ in range(count)]
    for index, label in enumerate(survivors):
        dealt[index % count].append(label)
    return Knockout(
        round=number,
        sheets=tuple(Sheet(number=n + 1, labels=tuple(one)) for n, one in enumerate(dealt)),
    )


def _keep(study: trials.Study, knockout: Knockout) -> Knockout:
    write_atomic(study.folder / SEARCHES / KNOCKOUT, knockout.model_dump_json(indent=2))
    return knockout


def places_of(profile: Profile) -> tuple[float, ...]:
    """The places along the line a sheet shows each variant at: grid.PLACES, evenly spread (each
    variant at its windows nearest them, as many as it has)."""
    xs = sorted(receiver.x for receiver in profile.receivers)
    span = xs[-1] - xs[0]
    return tuple(
        round(xs[0] + span * (2 * i + 1) / (2 * grid.PLACES), 1) for i in range(grid.PLACES)
    )


def first_sheets(rows: Sequence[Row], moved: Sequence[str]) -> tuple[Sheet, ...]:
    """The first round's sheets: SHEET variants each, of one window length, by length then by
    their settings' values in page order."""
    by_length: dict[int, list[Row]] = {}
    for row in rows:
        by_length.setdefault(row.length, []).append(row)
    labels: list[list[str]] = []
    for length in sorted(by_length):
        ordered = sorted(
            by_length[length],
            key=lambda row: tuple(row.values.get(name, "") for name in moved),
        )
        labels += [
            [row.label for row in ordered[i : i + SHEET]] for i in range(0, len(ordered), SHEET)
        ]
    return tuple(Sheet(number=n + 1, labels=tuple(one)) for n, one in enumerate(labels))


def really_bad(card: Mapping[str, Any]) -> bool:
    """A variant no sheet shows: no image on any window, or fewer than PASSING of its windows
    passing (no curve along the line)."""
    return card["image"] is None or card["covered"] < PASSING * card["windows"]


def axes(
    profile: Profile,
    mode: str,
    reach_m: float | None,
    cone: Cone | None,
    given: Given,
    grid_file: Grid | None = None,
) -> list[Axis]:
    """The settings the grid moves on `profile` in `mode`, each over the values the grid gives
    (grid.yaml's by default: the physics sets them), in PAC's page order; those the user gave left
    out."""
    grid_file = grid_file or grid_of()
    velocities = (
        float(grid_file.group_velocities.slowest.value),
        float(grid_file.group_velocities.fastest.value),
    )
    found: list[Axis] = []
    for name, setting in grid_file.method(mode).items():
        made = _values(name, setting, profile, reach_m, cone, velocities)
        found.append(Axis(name, made, SETS[name], setting.unit))
    return [axis for axis in found if axis.values and not _given(axis, given)]


def _values(
    name: str,
    setting: Setting,
    profile: Profile,
    reach_m: float | None,
    cone: Cone | None,
    velocities: tuple[float, float],
) -> tuple[Value, ...]:
    """Setting `name`'s values, as the line's settings take them."""
    match name:
        case "length":
            receivers = len(profile.receivers)
            longest = (
                max(int(receivers * setting.longest), min(setting.values))
                if setting.longest
                else receivers
            )
            return tuple(
                Value(str(n), n, {"masw": {"length": n}}) for n in setting.values if n <= longest
            )
        case "distance_min":
            return tuple(
                Value(f"{d:g} m" if d is not None else "none", d, {"masw": {"distance_min": d}})
                for d in setting.values
            )
        case "distance_max":
            xs = sorted(receiver.x for receiver in profile.receivers)
            reach = reach_m or xs[-1] - xs[0]
            return tuple(
                Value(
                    _reach_named(share),
                    round(share * reach, 1),
                    {"masw": {"distance_max": round(share * reach, 1)}},
                )
                for share in setting.values
            )
        case "near_field_m":
            return tuple(
                Value(_near_named(share), share, {NEAR: share}) for share in setting.values
            )
        case "muting":
            found: list[Value] = []
            for named in setting.values:
                stage = _mute(setting, named, cone, velocities, profile)
                if stage is not None:
                    found.append(Value(named, None, {"muting": stage}))
            return tuple(found)
        case "segments":
            return tuple(
                Value(f"{n:g} crossings", n, {SEGMENTS: n / velocities[0]}) for n in setting.values
            )
        case "fk_threshold":
            return tuple(
                Value(f"{t:g}", t, {"selection": {"method": "fk", "threshold": t}})
                for t in setting.values
            )
        case "fk_band":
            found = []
            named_bands: dict[str, tuple[float | None, float | None]] = {
                "open": (None, None),
                "group_velocities": velocities,
            }
            for band in setting.values:
                low, high = named_bands[band] if isinstance(band, str) else band
                named = band if isinstance(band, str) else f"{low}-{high} m/s"
                found.append(
                    Value(named, None, {"selection": {"method": "fk", "vmin": low, "vmax": high}})
                )
            return tuple(found)
        case _:  # a stage, as the preset takes it
            return tuple(
                Value(_stage_named(stage), None, {name: stage}) for stage in setting.values
            )


def _mute(
    setting: Setting,
    named: str,
    cone: Cone | None,
    velocities: tuple[float, float],
    profile: Profile,
) -> dict[str, Any] | None:
    """The muting stage of candidate `named`: none for no mute, as PACo's mute_candidates makes
    its own; None when it needs the survey's cone and there is none."""
    candidate = setting.candidates[named]
    if candidate is None:
        return {"method": "none"}
    known = {"slowest": velocities[0], "fastest": velocities[1]}
    if cone is not None:
        known |= {"slow": cone.vmin, "median": cone.median, "fast": cone.vmax}
    if any(one not in known for one in candidate.of):
        return None
    low, high = (known[one] for one in candidate.of)
    return {
        "method": "mute",
        "vmin": round(low / candidate.widen, 1),
        "vmax": round(high * candidate.widen, 1),
        "width": setting.parameters["width"].value,
        "taper": max(0, round(setting.parameters["taper"].value * profile.sampling_rate_hz)),
    }


def points(moved: Sequence[Axis]) -> list[dict[str, int]]:
    """Every combination of the settings' values: a point is a value's index by setting."""
    return [
        dict(zip((axis.name for axis in moved), indices, strict=True))
        for indices in product(*(range(len(axis.values)) for axis in moved))
    ]


def changes_of(
    moved: Sequence[Axis], point: Mapping[str, int], base: Mapping[str, Any], profile: Profile
) -> dict[str, Any] | None:
    """The changes of the line's settings `point` makes (`base`: the settings it changes, its
    length given): each of its values merged, a passive line's segments as long as its window
    says, no farthest shot nearer than the window's half length and two receivers, the near field
    a share of the window. None when it means nothing: a near field reaching the farthest shot."""
    made: dict[str, Any] = {}
    for axis in moved:
        if axis.name in point:
            made = deep_merge(made, axis.values[point[axis.name]].changes)
    masw = {**base.get("masw", {}), **made.get("masw", {})}
    spacing = _spacing(profile)
    span = (int(masw["length"]) - 1) * spacing
    per_metre = made.pop(SEGMENTS, None)
    if per_metre is not None:
        seconds = round(per_metre * span, 2)
        made["slicing"] = {"segment_duration": seconds, "segment_step": seconds}
    farthest = masw.get("distance_max")
    if farthest is not None and "masw" in made:
        nearest = round(span / 2 + 2 * spacing, 1)
        if float(farthest) < nearest:
            made["masw"] = {**made["masw"], "distance_max": nearest}
            farthest = nearest
    share = made.pop(NEAR, None)
    if share is not None:
        near = round(share * span, 1)
        if near and farthest is not None and near >= float(farthest):
            return None
        made[trials.NEAR_FIELD] = near or None
    return made


def search(
    study: trials.Study,
    *,
    line: trials.LineState | None,
    settings: Settings,
    config: QCConfig,
    windows: str = WINDOWS,
    grid_changes: Mapping[str, Any] | None = None,
    on_progress: trials.Progress | None = None,
) -> tuple[Search, Knockout | None]:
    """The full grid of the line's settings on the trial set `windows`, from the preset and the
    user's settings, each variant scored by the grid, its thumbnails kept; `grid_changes`: of
    grid.yaml's values, for this study. The really bad dropped, the others sorted into the first
    round's sheets (none when no variant gave an image to judge). Once a study, saved in its
    searches/; the scoring said to `on_progress` as it goes."""
    trials.check_open(study)
    if searched(study) is not None:
        raise ValueError("The grid has run: search shows its sheets left to judge.")
    started = datetime.now(UTC)
    profile = load_profile(study.profile, settings)
    user = given_of(study)
    base = with_line({}, user)
    # The survey first (P1 makes it): its reach and cone set the grid's distances and mutes.
    found = surveying.survey(study, settings, config)
    reach = trials.survey_reach(study)
    cone = (
        Cone(found.cone_m_s[0], found.cone_m_s[2], found.cone_m_s[1], found.cone_traces or 0)
        if found.cone_m_s is not None
        else None
    )
    mode = mode_of(study, profile)
    try:
        grid_file = grid_of(grid_changes)
    except ValueError as error:
        raise ValueError(f"grid: {error}") from None
    moved = axes(profile, mode, reach, cone, user, grid_file)
    variants: dict[str, trials.Trial] = {}
    changes: dict[str, dict[str, Any]] = {}
    values: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    refused: list[str] = []
    for point in points(moved):
        change = changes_of(moved, point, base, profile)
        if change is None:
            continue
        try:
            check_line(change, user)
            trial = trials.concrete(deep_merge(dict(base), change), profile, config, reach)
        except ValueError as error:
            refused.append(str(error))
            continue
        same = json.dumps(
            [trial.preset.model_dump(mode="json"), trial.near_field_m], sort_keys=True
        )
        if same in seen:
            continue
        seen.add(same)
        label = f"g{len(variants) + 1}"
        variants[label] = trial
        changes[label] = change
        values[label] = {axis.name: axis.values[point[axis.name]].name for axis in moved}
    folder = study.folder / SEARCHES
    images = folder / IMAGES
    places = places_of(profile)
    runs, failed = grid.score(
        study, variants, windows, line, profile, settings, config, images, places, on_progress
    )
    refused += failed.values()
    rows = [
        Row(
            label=label,
            values=values[label],
            changes=changes[label],
            trial=run.key,
            length=run.length,
            card=_card(run),
        )
        for label, run in runs.items()
    ]
    shown = [row for row in rows if not really_bad(row.card)]
    for row in rows:
        if really_bad(row.card):
            (images / f"{row.label}.npz").unlink(missing_ok=True)
    made = Search(
        id=SEARCH,
        windows=windows,
        axes={axis.name: [value.name for value in axis.values] for axis in moved},
        units={axis.name: axis.unit for axis in moved},
        changes=dict(grid_changes or {}),
        rows=tuple(rows),
        refused=len(refused),
        refusal=refused[0] if refused else None,
        places=places,
        dropped=len(rows) - len(shown),
        seconds=round((datetime.now(UTC) - started).total_seconds(), 1),
        made_at=started,
    )
    folder.mkdir(exist_ok=True)
    write_atomic(folder / f"{made.id}.json", made.model_dump_json(indent=2))
    if not shown:
        return made, None
    labels = [row.label for row in shown]
    knockout = (
        Knockout(round=1, sheets=first_sheets(shown, [axis.name for axis in moved]))
        if len(labels) > KEEP
        else next_round(1, labels)
    )
    return made, _keep(study, knockout)


def mode_of(study: trials.Study, profile: Profile) -> str:
    """The line's method: the mode the user gave, else its profile's kind."""
    return str(with_line({}, given_of(study)).get(trials.MODE) or profile.kind)


def row_changes(study: trials.Study, label: str) -> dict[str, Any]:
    """The changes of the search's variant `label`, as try_settings and commit take them."""
    made = searched(study)
    if made is None:
        raise ValueError("No search yet: run search first.")
    row = next((row for row in made.rows if row.label == label), None)
    if row is None:
        raise ValueError(f"grid: no variant {label} in {made.id}.")
    return dict(row.changes)


def describe(made: Search, knockout: Knockout | None) -> str:
    """The search in words: the grid it ran, the variants it dropped and how the others are
    judged, sheet by sheet; no score."""
    moved = {name: values for name, values in made.axes.items() if len(values) > 1}
    grid_said = "; ".join(
        f"{name} ({made.units[name]}) {', '.join(values)}" for name, values in moved.items()
    )
    fixed = "; ".join(
        f"{name} {values[0]}" for name, values in made.axes.items() if len(values) == 1
    )
    lines = [
        f"{made.id}: the full grid, {len(made.rows)} variants on {made.windows} in "
        f"{made.seconds / 60:.0f} min, every combination of: {grid_said}."
        + (f" The same in every variant: {fixed}." if fixed else "")
        + (f" {made.refused} could not be tried: {made.refusal}" if made.refused else "")
    ]
    if knockout is None:
        lines.append(
            "No variant gave an image passing enough windows to judge: look at the survey's "
            "records, and try settings with try_settings."
        )
        return "\n".join(lines)
    at = ", ".join(f"{place:g}" for place in made.places)
    lines.append(
        f"{made.dropped} dropped as really bad: no image, or fewer than {PASSING:.0%} of their "
        f"windows passing (no curve along the line). The other {len(made.rows) - made.dropped} "
        f"are in sheets of up to {SHEET}, by window length first, one variant a row at its windows "
        f"nearest {len(made.places)} places along the line ({at} m), judged by eye with no score: "
        f"each sheet's {KEEP} cleanest kept and mixed into new sheets, round after round, until "
        f"{KEEP} are left, whose sheet comes to you: choose the cleanest (choose), the search's "
        "choice, and commit it. Nothing is tried after it: every combination was seen."
    )
    return "\n".join(lines)


def sheet_text(made: Search, knockout: Knockout, sheet: Sheet) -> str:
    """A sheet in words: where it is in the knockout, then each of its variants' settings in
    reading order (one a row, top to bottom)."""
    moved = {name for name, values in made.axes.items() if len(values) > 1}
    by_label = {row.label: row for row in made.rows}
    places = ", ".join(f"{place:g}" for place in made.places)
    if knockout.final:
        head = (
            f"The final: the {len(sheet.labels)} cleanest of the search, one a row, each at its "
            f"windows nearest {places} m. Choose the cleanest by eye (choose), with its reason, "
            "then commit it:"
        )
    else:
        head = (
            f"Round {knockout.round}, sheet {sheet.number} of {len(knockout.sheets)} (keep "
            f"{to_keep(knockout, sheet)}): {len(sheet.labels)} variants, one a row, each at its "
            f"windows nearest {places} m:"
        )
    return "\n".join(
        [head]
        + [
            f"- {label}: {_named(by_label[label], moved) or 'the one variant'}"
            for label in sheet.labels
        ]
    )


def describe_row(made: Search, row: Row) -> str:
    """A variant's settings in words: its values of the settings the grid moved, the numbers its
    window gives them."""
    moved = {name for name, values in made.axes.items() if len(values) > 1}
    return _named(row, moved) or "the one variant"


def _named(row: Row, moved: set[str]) -> str:
    """A variant's values of the settings `moved` in words, the numbers its window gives them."""
    said = []
    for name, value in row.values.items():
        if name not in moved:
            continue
        if name == "segments":
            value = f"{row.changes['slicing']['segment_duration']:g} s ({value})"
        elif name == "distance_max":
            value = f"{row.changes['masw']['distance_max']:g} m ({value})"
        elif name == "near_field_m" and (near := row.changes.get(trials.NEAR_FIELD)):
            value = f"{near:g} m ({value})"
        said.append(f"{name} {value}")
    return " + ".join(said)


def _card(run: trials.TrialRun) -> dict[str, Any]:
    card = trials.card_of(run)
    # The band the pick holds along the line: each window's, none where it passes no curve.
    bands = [
        high - low
        if window.g3 is not None and window.g3.verdict == "pass" and window.g3.kept.band_hz
        else 0.0
        for window in run.windows
        for low, high in (
            window.g3.kept.band_hz if window.g3 is not None and window.g3.kept.band_hz else (0, 0),
        )
    ]
    return {
        "covered": len(card.covered),
        "windows": len(card.windows),
        # Windows its FK selection left without a segment, and so without an image.
        "empty": sum(1 for window in run.windows if window.error and KEPT_NONE in window.error),
        "image": card.image,
        "image_p10": card.image_p10,
        "stripes": card.stripes,
        "band_hz": round(sum(bands) / len(bands), 1) if bands else None,
        "depth_m": card.depth_m,
        "depth_p10_m": card.depth_p10_m,
        "beyond": card.beyond,
    }


def _spacing(profile: Profile) -> float:
    xs = sorted(receiver.x for receiver in profile.receivers)
    gaps = [b - a for a, b in pairwise(xs) if b > a]
    return min(gaps) if gaps else 1.0


# The settings an image's costly part depends on, by mode (the grid's bases, grid.base_of).
BASES = {
    "active": ("length", "trigger", "muting", "filtering", "dispersion"),
    "passive-active": ("length", "trigger", "muting", "filtering", "dispersion"),
    "passive": ("length", "filtering", "segments", "whitening", "normalization", "dispersion"),
}
# A cone standing in for a survey's when sizing a line's grid: every mute candidate counted.
SIZING_CONE = Cone(vmin=100.0, vmax=1000.0, median=300.0, n_traces=1)


def size(
    mode: str, profile: Profile | None = None, grid_file: Grid | None = None
) -> tuple[int, int]:
    """The grid's variants and bases in `mode`: every combination of its values; on `profile`, as
    its search makes them (the lengths it allows, no near field reaching the farthest shot, no
    two alike), every mute counted, the reach the line's length."""
    grid_file = grid_file or grid_of()
    if profile is None:
        settings = grid_file.method(mode)
        variants = bases = 1
        for name, setting in settings.items():
            variants *= len(setting.values)
            bases *= len(setting.values) if name in BASES[mode] else 1
        return variants, bases
    moved = axes(profile, mode, None, SIZING_CONE, Given(), grid_file)
    made: set[str] = set()
    based: set[str] = set()
    for point in points(moved):
        change = changes_of(moved, point, {}, profile)
        if change is None:
            continue
        made.add(json.dumps(change, sort_keys=True))
        stages = ["slicing" if name == "segments" else name for name in BASES[mode]]
        held = [change["masw"]["length"], *(change.get(stage) for stage in stages[1:])]
        based.add(json.dumps(held, sort_keys=True))
    return len(made), len(based)


def main() -> None:
    """paco-grid: the size of the search's grid, from grid.yaml: by method, each setting's values
    and the variants so far, then the total; on each profile given, as its search makes them."""
    parser = argparse.ArgumentParser(prog="paco-grid", description=main.__doc__)
    parser.add_argument(
        "profiles", nargs="*", help="profiles to size it on, each in its modes (PACO_INPUT_DIR)"
    )
    arguments = parser.parse_args()
    grid_file = grid_of()
    print(files("paco.autonomous") / GRID_FILE)
    for mode in BASES:
        settings = grid_file.method(mode)
        counts = [(name, setting.unit, len(setting.values)) for name, setting in settings.items()]
        print(_table(mode, counts, *size(mode, grid_file=grid_file)))
    for name in arguments.profiles:
        profile = load_profile(name, get_settings())
        for mode in MODES[profile.kind]:
            moved = axes(profile, str(mode), None, SIZING_CONE, Given(), grid_file)
            counts = [(axis.name, axis.unit, len(axis.values)) for axis in moved]
            title = f"{name} ({mode}: its lengths, no near field reaching the farthest shot)"
            print(_table(title, counts, *size(str(mode), profile, grid_file)))


def _table(title: str, counts: Sequence[tuple[str, str, int]], variants: int, bases: int) -> str:
    """A grid's size as a table: each setting, its unit, its values and the variants so far; then
    the total."""
    width = max(len(name) for name, _, _ in counts)
    unit = max(len(one) for _, one, _ in counts)
    lines = [
        f"\n{title}",
        f"  {'setting':<{width}}  {'unit':<{unit}}  {'values':>6}  {'so far':>8}",
    ]
    product = 1
    for name, one, number in counts:
        product *= number
        lines.append(f"  {name:<{width}}  {one:<{unit}}  {number:>6}  {product:>8}")
    lines.append(
        f"  {'total':<{width}}  {'':<{unit}}  {'':>6}  {variants:>8} variants, {bases} bases"
    )
    return "\n".join(lines)


def _reach_named(share: float) -> str:
    """A share of the reach in words: the reach, reach/2, 0.75 reach."""
    if share == 1:
        return "the reach"
    parts = 1 / share
    return f"reach/{parts:g}" if parts == round(parts) else f"{share:g} reach"


def _near_named(share: float) -> str:
    """A share of the window in words: none, half the window, the window, 0.25 window."""
    return {0.0: "none", 0.5: "half the window", 1.0: "the window"}.get(share, f"{share:g} window")


def _stage_named(stage: Mapping[str, Any]) -> str:
    """A stage in words: linear, root 2, phase_weighted nu 2, fmin 0 fmax 100."""
    said = [str(stage["method"])] if "method" in stage else []
    said += [
        str(value) if key == "n" else f"{key} {value}"
        for key, value in stage.items()
        if key != "method"
    ]
    return " ".join(said)


def _given(axis: Axis, given: Given) -> bool:
    """Whether the user gave a setting `axis` sets: it stays as they gave it."""
    for stage, key in axis.sets:
        if stage == "near_field_m" or stage not in given.line:
            continue
        if key == "*" or key in given.line[stage]:
            return True
    return False


if __name__ == "__main__":
    main()
