"""search: the full grid of the line's settings, scored in code, for the model to choose from (the
user, 2026-10-20: PACo's workflow is one thing; the autonomous agent runs a full grid, then the
model chooses the best results and checks that they make sense). Every setting of PAC's page for
the line's method takes values the physics sets (axes), and every combination of them is a
variant: the windows' length over the line's range; the farthest shot as shares of the survey's
reach (the surface waves fall into the noise past it), never nearer than the window's half
length; the near field as shares of the window (a shot nearer than half a wavelength bends the
low frequencies); PACo's mutes from the survey's cone; a passive line's segments as crossings of
the window by the slowest wave, PACo's FK thresholds and bands, the whitening and the
normalization. Each variant is judged on the same trial windows as try_settings judges it, by the
grid (paco.autonomous.tools.grid): the scores a trial gives. No score reads an image's quality
as the eye does (their background favours long windows, the windows passing short ones): the
shortlist holds the best by each of several measures, a variety of the windows' settings, an
experiment the model looks at and judges by eye; the summary by setting shows where the grid is
best.

A study's search: searches/S1.json, each variant with its settings, trial and scorecard."""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
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
WINDOWS = "spread:12"
SHORTLIST = trials.MOST_VARIANTS
# The grid's values, a file beside the code (the user, 2026-10-20: every grid value in one file).
GRID_FILE = "grid.yaml"
# The pseudo-settings a value may give, resolved by each variant's window (changes_of): a passive
# line's segments, their seconds per metre of the window; the near field, its share of the window.
SEGMENTS = "segment_s_per_m"
NEAR = "near_field_window"
# The shortlist: the best PER_MEASURE at least by each measure; a variant counts for a measure other
# than the windows passing with this share of its windows passing (a smear without a ridge has no
# stripes); no two alike in these settings, which make an image what it is.
PER_MEASURE = 3
PASSING = 0.5
DISTINCT = ("length", "distance_max", "muting", "segments", "whitening", "normalization")
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
    many could not be and why the first could not, the shortlist (its rows' labels, each with the
    measures it leads) and its experiment."""

    model_config = ConfigDict(frozen=True)

    id: str
    windows: str
    axes: dict[str, list[str]]
    units: dict[str, str]
    changes: dict[str, Any]  # of grid.yaml's values, for this study
    rows: tuple[Row, ...]
    refused: int
    refusal: str | None
    shortlist: dict[str, list[str]]
    experiment: str
    seconds: float
    made_at: datetime


def searched(study: trials.Study) -> Search | None:
    """The study's search, when it has run."""
    path = study.folder / SEARCHES / f"{SEARCH}.json"
    return Search.model_validate_json(path.read_text()) if path.exists() else None


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
) -> tuple[Search, trials.Experiment]:
    """The full grid of the line's settings on the trial set `windows`, from the preset and the
    user's settings, each variant scored by the grid; `grid_changes`: of grid.yaml's values, for
    this study. The shortlist tried as an experiment. Once a study, saved in its searches/."""
    trials.check_open(study)
    if (made := searched(study)) is not None:
        raise ValueError(
            f"The grid has run: {made.id}, its shortlist {made.experiment}. Refine around your "
            "choice with try_settings."
        )
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
    mode = str(base.get(trials.MODE) or profile.kind)
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
    runs, failed = grid.score(study, variants, windows, line, profile, settings, config)
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
    shortlist = _shortlist(rows)
    by_label = {row.label: row for row in rows}
    experiment = trials.try_settings(
        study,
        [by_label[label].changes for label in shortlist],
        windows,
        current=None,
        line=line,
        settings=settings,
        config=config,
    )
    made = Search(
        id=SEARCH,
        windows=windows,
        axes={axis.name: [value.name for value in axis.values] for axis in moved},
        units={axis.name: axis.unit for axis in moved},
        changes=dict(grid_changes or {}),
        rows=tuple(rows),
        refused=len(refused),
        refusal=refused[0] if refused else None,
        shortlist=shortlist,
        experiment=experiment.id,
        seconds=round((datetime.now(UTC) - started).total_seconds(), 1),
        made_at=started,
    )
    folder = study.folder / SEARCHES
    folder.mkdir(exist_ok=True)
    write_atomic(folder / f"{made.id}.json", made.model_dump_json(indent=2))
    return made, experiment


# A page of the search's table, rows.
PAGE = 30


def table(
    made: Search,
    sort: str | None = None,
    where: Mapping[str, Sequence[str]] | None = None,
    page: int = 1,
) -> str:
    """A page of the search's table: its variants with the values `where` keeps of each setting
    (by the names the table gives), best first by the measure `sort` (one of MEASURES' names; the
    image score by default), each with its label, its settings and its scores."""
    measures = dict(MEASURES)
    named = sort or MEASURES[0][0]
    if named not in measures:
        raise ValueError(f"sort: one of {', '.join(measures)}, not {named}.")
    for name, values in (where or {}).items():
        if name not in made.axes:
            raise ValueError(f"where: no setting {name}; the grid's: {', '.join(made.axes)}.")
        if lost := sorted(set(values) - set(made.axes[name])):
            raise ValueError(
                f"where {name}: no value {', '.join(lost)}; its: {', '.join(made.axes[name])}."
            )
    value = measures[named]
    kept = [
        row
        for row in made.rows
        if all(row.values.get(name) in values for name, values in (where or {}).items())
    ]
    kept.sort(key=lambda row: -_known(value(row.card)))
    pages = max(1, -(-len(kept) // PAGE))
    if not 1 <= page <= pages:
        raise ValueError(f"page: 1 to {pages}.")
    moved = {name for name, values in made.axes.items() if len(values) > 1}
    lines = [
        f"{made.id}: {len(kept)} of its {len(made.rows)} variants, best first by {named}, page "
        f'{page} of {pages}. try_settings takes a row as {{"grid": "<label>"}}:'
    ]
    for row in kept[(page - 1) * PAGE : page * PAGE]:
        lines.append(f"- {row.label}: {_named(row, moved)}: {_scored(row.card)}")
    return "\n".join(lines)


def _known(value: float | None) -> float:
    """A measure's value, the worst when unknown."""
    return value if value is not None else float("-inf")


def row_changes(study: trials.Study, label: str) -> dict[str, Any]:
    """The changes of the search's variant `label`, as try_settings takes them."""
    made = searched(study)
    if made is None:
        raise ValueError("No search yet: run search first.")
    row = next((row for row in made.rows if row.label == label), None)
    if row is None:
        raise ValueError(f"grid: no variant {label} in {made.id}.")
    return dict(row.changes)


def describe(made: Search, experiment: trials.Experiment) -> str:
    """The search in words: the grid it ran, its shortlist (the experiment, each variant with the
    measures it is among the best by), then each setting's values, the best each gave."""
    labels = dict(zip(made.shortlist, (one.label for one in experiment.tried[1:]), strict=True))
    by_label = {row.label: row for row in made.rows}
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
        + (f" {made.refused} could not be tried: {made.refusal}" if made.refused else ""),
        f"The shortlist is experiment {experiment.id}: the best {PER_MEASURE} at least by each "
        "measure (the image score, the fewest stripes, the windows passing, the widest band the "
        "pick holds along the line, the depth of the weakest windows), no two alike in their windows' "
        "settings. "
        "None of them reads an image's quality: judge by eye, and check that your choice makes "
        "sense. The shortlist, the measures each is among the best by:",
    ]
    for label, measures in made.shortlist.items():
        row = by_label[label]
        lines.append(
            f"- {experiment.id}:{labels[label]} ({', '.join(measures)}) {label}: "
            f"{_named(row, set(moved))}: {_scored(row.card)}"
        )
    lines.append(
        f"By setting, each value's best image score and widest band among its variants passing "
        f"{PASSING:.0%} of their windows, and how many do:"
    )
    for name, names in moved.items():
        said = []
        for value in names:
            having = [row for row in made.rows if row.values.get(name) == value]
            passing = [row for row in having if _passing(row.card) >= PASSING]
            images = [row.card["image"] for row in passing if row.card["image"] is not None]
            bands = [row.card["band_hz"] for row in passing if row.card["band_hz"] is not None]
            if images and bands:
                best = f"{max(images):.2f}, {max(bands):.0f} Hz"
            elif having and all(row.card["empty"] == row.card["windows"] for row in having):
                best = "its FK selection keeps no segment"
            else:
                best = "none passing"
            said.append(f"{value}: {best} ({len(passing)}/{len(having)})")
        lines.append(f"- {name}: " + "; ".join(said))
    return "\n".join(lines)


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


MEASURES: tuple[tuple[str, Callable[[Mapping[str, Any]], float | None]], ...] = (
    ("image score", lambda card: card["image"]),
    ("fewest stripes", lambda card: -card["stripes"] if card["stripes"] is not None else None),
    (
        "windows passing",
        lambda card: card["covered"] / card["windows"] if card["windows"] else None,
    ),
    ("widest band along the line", lambda card: card["band_hz"]),
    ("depth of the weakest", lambda card: card["depth_p10_m"]),
)


def _shortlist(rows: Sequence[Row]) -> dict[str, list[str]]:
    """The best by each measure in turn, PER_MEASURE of each at least, then deeper while there is
    room, SHORTLIST at most, none alike another in the DISTINCT settings: each row's label with the
    measures it is among the best by, by window length."""
    rankings: list[tuple[str, list[Row]]] = []
    for name, value in MEASURES:
        counted = [
            row
            for row in rows
            if value(row.card) is not None
            and (name == "windows passing" or _passing(row.card) >= PASSING)
        ]
        # Ties to the band along the line: the ridge unbroken over a wide band comes first.
        rankings.append(
            (
                name,
                sorted(
                    counted,
                    key=lambda row: (-(value(row.card) or 0.0), -(row.card["band_hz"] or 0.0)),
                ),
            )
        )
    chosen: dict[str, list[str]] = {}
    held: dict[tuple[str | None, ...], str] = {}
    turn = 0
    while True:
        added = False
        for name, ranking in rankings:
            while ranking:
                row = ranking.pop(0)
                alike = tuple(row.values.get(setting) for setting in DISTINCT)
                if row.label in chosen:
                    chosen[row.label].append(name)
                elif alike in held:
                    continue
                elif len(chosen) < SHORTLIST:
                    chosen[row.label] = [name]
                    held[alike] = row.label
                else:
                    break
                added = True
                break
        turn += 1
        if not added or (len(chosen) >= SHORTLIST and turn >= PER_MEASURE):
            break
    order = {row.label: (row.length, -(row.card["image"] or 0.0)) for row in rows}
    return dict(sorted(chosen.items(), key=lambda item: order[item[0]]))


def _passing(card: Mapping[str, Any]) -> float:
    return card["covered"] / card["windows"] if card["windows"] else 0.0


def _scored(card: Mapping[str, Any]) -> str:
    def part(name: str, spec: str) -> str:
        value = card.get(name)
        return format(value, spec) if value is not None else "-"

    return (
        f"{card['covered']}/{card['windows']} pass, image {part('image', '.2f')} (P10 "
        f"{part('image_p10', '.2f')}), stripes {part('stripes', '.3f')}, band "
        f"{part('band_hz', '.0f')} Hz, depth {part('depth_m', '.0f')} m, beyond reach "
        f"{part('beyond', '.0%')}"
    )


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
