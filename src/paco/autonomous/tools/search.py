"""search: the line's settings tried on a grid in code, fast, for the model to judge and refine (the
user, 2026-10-20: one side PACo's workflow, the other the autonomous agent, a search a model judges
and refines). Every setting of the line's method takes a grid of values over its whole range
(axes): each value tried alone from the current settings, then the best of those changes together,
in pairs and all at once, each variant on the same trial windows as try_settings tries it, several
at once on the machine's workers. A full grid would hold millions of variants, and one sampled at
random would carry a poor value of some setting in nearly every variant. No score reads an image's
quality as the eye does (their background favours long windows, the windows passing short ones):
the shortlist holds the best by each of several measures, an experiment the model looks at and
judges by eye, with the table of every variant tried.

A study's searches: searches/S<n>.json, each variant with its settings, trial and scorecard."""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations, pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict
from sigpipe.masw.profiles import Profile, load_profile
from sigpipe.masw.quality.cone import Cone
from sigpipe.masw.runs.writing import write_atomic

from paco import stopping
from paco.autonomous.tools import survey as surveying
from paco.autonomous.tools import trials
from paco.autonomous.tools.given import Given, check_line, given_of, with_line
from paco.qc.config import QCConfig
from paco.qc.loops import deep_merge
from paco.qc.muting import mute_candidates
from paco.settings import Settings

SEARCHES = "searches"  # in the study's folder
WINDOWS = "spread:12"
SHORTLIST = trials.MOST_VARIANTS
CURRENT = "current"  # the current settings' row
# The best changes alone that the second round combines, in pairs and all at once.
COMBINED = 8
# The window lengths tried, receivers: every one up to 12, then further apart.
LENGTHS = (*range(5, 13), 14, 16, 18, 20, 24, 28, 32, 40, 48)
# The measures the shortlist takes the best of, each its name and its value on a scorecard (higher
# better): they pull different ways (the image score and the depth to long windows, the windows
# passing to short ones), so that together they span the lengths.
PER_MEASURE = 3
# A variant counts for a measure other than the windows passing with this share of its windows
# passing: a smear without a ridge has no stripes.
PASSING = 0.5
# A passive line's segments, end to end, in times the slowest wave (G1's vg_min) takes to cross the
# window, √2 apart: short, as noise segments work best (the user, 2026-10-20: 1 s or less, by the
# window's length and the velocities), where PACo's segment trial tries 10 to 80.
CROSSINGS = (2, 3, 4, 6, 8, 11, 16)
# The segments' pseudo-setting: their seconds per metre of the window, each variant's length
# setting their seconds.
SEGMENTS = "segment_s_per_m"


@dataclass(frozen=True)
class Value:
    """A setting's value: its name, its number when it has one (a range keeps it by it), and the
    changes of the line's settings it makes."""

    name: str
    number: float | None
    changes: Mapping[str, Any]


@dataclass(frozen=True)
class Axis:
    """A setting the search moves, its values in order (a step moves to the next one), and the
    line's settings it sets, by stage and key ("*": the whole stage)."""

    name: str
    values: tuple[Value, ...]
    sets: tuple[tuple[str, str], ...]


class Row(BaseModel):
    """A variant the search tried: its value by axis, its changes, its trial and its scorecard."""

    model_config = ConfigDict(frozen=True)

    label: str  # s<n>
    values: dict[str, str]  # the settings it changes, each its value
    changes: dict[str, Any]
    trial: str
    length: int
    card: dict[str, Any]


class Search(BaseModel):
    """A search: its trial set, the current settings, every variant tried, why some could not be,
    the shortlist (its rows' labels) and its experiment."""

    model_config = ConfigDict(frozen=True)

    id: str  # S<n>, in its study
    windows: str
    current: dict[str, Any]
    rows: tuple[Row, ...]
    refused: tuple[str, ...]
    shortlist: dict[str, list[str]]  # each row's label, and the measures it leads
    experiment: str
    seconds: float
    made_at: datetime


def axes(
    profile: Profile,
    mode: str,
    reach_m: float | None,
    cone: Cone | None,
    config: QCConfig,
    given: Given,
    ranges: Mapping[str, Sequence[Any]] | None = None,
) -> list[Axis]:
    """The settings the search moves on `profile` in `mode`, each over its whole range; those the
    user gave left out, each narrowed as `ranges` says ([low, high] for a number, the names kept
    for a choice)."""
    longest = max(min(len(profile.receivers) // 2, 48), 5)
    found = [
        Axis(
            "length",
            tuple(Value(str(n), n, {"masw": {"length": n}}) for n in LENGTHS if n <= longest),
            (("masw", "length"),),
        )
    ]
    if mode != "passive":
        xs = sorted(receiver.x for receiver in profile.receivers)
        reached = [float(round(reach_m))] if reach_m else []
        distances = sorted({*_spread(10.0, xs[-1] - xs[0], 9), *reached})
        found += [
            Axis(
                "distance_max",
                tuple(Value(f"{d:g} m", d, {"masw": {"distance_max": d}}) for d in distances),
                (("masw", "distance_max"),),
            ),
            Axis("muting", _mutes(profile, cone, config), (("muting", "*"),)),
        ]
    if mode == "active":
        found += [
            Axis(
                "near_field_m",
                tuple(
                    Value(f"{d:g} m" if d else "none", d, {"near_field_m": d or None})
                    for d in (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)
                ),
                (("near_field_m", "*"),),
            ),
            Axis(
                "image_stacking",
                (
                    Value("linear", None, {"image_stacking": {"method": "linear"}}),
                    Value("root 2", None, {"image_stacking": {"method": "root", "n": 2}}),
                    Value("root 3", None, {"image_stacking": {"method": "root", "n": 3}}),
                ),
                (("image_stacking", "*"),),
            ),
        ]
    if mode == "passive":
        found += _interferometry(config)
    if mode != "active":
        found.append(Axis("stacking", _stackings(), (("stacking", "*"),)))
    unknown = sorted(set(ranges or {}) - {axis.name for axis in found})
    if unknown:
        raise ValueError(
            f"ranges: no setting {', '.join(unknown)} on this line; its settings: "
            f"{', '.join(axis.name for axis in found)}."
        )
    return [
        _narrowed(axis, (ranges or {}).get(axis.name)) for axis in found if not _given(axis, given)
    ]


def grid(axes: Sequence[Axis]) -> list[dict[str, int]]:
    """Each value of each axis alone, the others left as the current settings have them: a point
    is a value's index by axis."""
    return [{axis.name: index} for axis in axes for index in range(len(axis.values))]


def combined(rows: Sequence[Row], moved: Sequence[Axis], current: Row) -> list[dict[str, int]]:
    """The best value of each setting but the length, among those that beat the current settings'
    image score or stripes losing at most a window, COMBINED at most: each pair of them and all at
    once, at the current length; and all at once at each length the shortlist holds."""
    best: list[tuple[float, dict[str, int]]] = []
    for axis in moved:
        if axis.name == "length":
            continue
        beating = [
            row
            for row in rows
            if set(row.values) == {axis.name}
            and row.card["covered"] >= current.card["covered"] - 1
            and _beats(row.card, current.card)
        ]
        if beating:
            row = min(beating, key=lambda one: _ranks(one.card, beating))
            best.append((_ranks(row.card, beating), _point(moved, row)))
    chosen = [point for _, point in sorted(best, key=lambda one: one[0])[:COMBINED]]
    together = [{**one, **other} for one, other in combinations(chosen, 2)]
    if len(chosen) > 2:
        everything = {name: index for point in chosen for name, index in point.items()}
        together.append(everything)
        lengths = {
            row.values["length"]
            for label in _shortlist(rows)
            for row in rows
            if row.label == label and set(row.values) == {"length"}
        }
        axis = next((one for one in moved if one.name == "length"), None)
        if axis is not None:
            together += [
                {**everything, "length": i}
                for i, value in enumerate(axis.values)
                if value.name in lengths
            ]
    return together


def _beats(card: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Whether a variant's image score or stripes beat the current settings'."""
    image, stripes = card["image"], card["stripes"]
    return (image is not None and image > (current["image"] or 0.0)) or (
        stripes is not None
        and stripes < (current["stripes"] if current["stripes"] is not None else 1.0)
    )


def _ranks(card: Mapping[str, Any], rows: Sequence[Row]) -> float:
    """A variant's rank among `rows` by its image score plus its rank by its stripes."""
    images = sorted(-(one.card["image"] or 0.0) for one in rows)
    stripes = sorted(
        (one.card["stripes"] if one.card["stripes"] is not None else 1.0) for one in rows
    )
    mine = card["stripes"] if card["stripes"] is not None else 1.0
    return images.index(-(card["image"] or 0.0)) + stripes.index(mine)


def changes_of(
    axes: Sequence[Axis], point: Mapping[str, int], base: Mapping[str, Any], profile: Profile
) -> dict[str, Any]:
    """The changes of the line's settings `point` makes (`base` its current settings, its length
    given): each of its values merged, a passive line's segments as long as its window says, and
    no farthest shot nearer than the window's half length and two receivers."""
    made: dict[str, Any] = {}
    for axis in axes:
        if axis.name in point:
            made = deep_merge(made, axis.values[point[axis.name]].changes)
    masw = {**base.get("masw", {}), **made.get("masw", {})}
    spacing = _spacing(profile)
    per_metre = made.pop(SEGMENTS, None)
    if per_metre is not None:
        seconds = round(per_metre * (int(masw["length"]) - 1) * spacing, 2)
        made["slicing"] = {"segment_duration": seconds, "segment_step": seconds}
    if masw.get("distance_max") is not None and ("masw" in made):
        nearest = (int(masw["length"]) - 1) * spacing / 2 + 2 * spacing
        if float(masw["distance_max"]) < nearest:
            made["masw"] = {**made["masw"], "distance_max": round(nearest, 1)}
    return made


def search(
    study: trials.Study,
    *,
    current: Mapping[str, Any] | None,
    line: trials.LineState | None,
    settings: Settings,
    config: QCConfig,
    windows: str = WINDOWS,
    ranges: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[Search, trials.Experiment]:
    """The line's settings tried on the trial set `windows`: each value of every setting alone
    from the current settings, then the best of those changes together; the shortlist tried as an
    experiment (its trials made already). Saved in the study's searches/."""
    trials.check_open(study)
    started = datetime.now(UTC)
    profile = load_profile(study.profile, settings)
    user = given_of(study)
    reach = trials.survey_reach(study)
    base = with_line(current or {}, user)
    # The current length given, which each variant's segments and nearest farthest shot follow.
    length = trials.concrete(base, profile, config, reach).preset.masw.length
    base = deep_merge(base, {"masw": {"length": length}})
    found = surveying.survey(study, settings, config)
    cone = (
        Cone(found.cone_m_s[0], found.cone_m_s[2], found.cone_m_s[1], found.cone_traces or 0)
        if found.cone_m_s is not None
        else None
    )
    mode = str(base.get(trials.MODE) or profile.kind)
    moved = axes(profile, mode, reach, cone, config, user, ranges)
    folder = study.folder / SEARCHES
    folder.mkdir(exist_ok=True)
    name = f"S{len(list(folder.glob('S*.json'))) + 1}"
    trying = _Trying(study, moved, base, user, profile, line, windows, settings, config)
    trying.run([{}, *grid(moved)])
    baseline = trying.rows.pop(CURRENT)
    trying.run(combined(list(trying.rows.values()), moved, baseline))
    shortlist = _shortlist(list(trying.rows.values()))
    if not shortlist:
        raise ValueError(f"No variant could be tried: {'; '.join(trying.refused[:3])}.")
    experiment = trials.try_settings(
        study,
        [trying.rows[label].changes for label in shortlist],
        windows,
        current=current,
        line=line,
        settings=settings,
        config=config,
    )
    made = Search(
        id=name,
        windows=windows,
        current=dict(current or {}),
        rows=(baseline, *trying.rows.values()),
        refused=tuple(trying.refused),
        shortlist=shortlist,
        experiment=experiment.id,
        seconds=round((datetime.now(UTC) - started).total_seconds(), 1),
        made_at=started,
    )
    write_atomic(folder / f"{made.id}.json", made.model_dump_json(indent=2))
    return made, experiment


def describe(made: Search, experiment: trials.Experiment) -> str:
    """The search in words: what it tried, the shortlist's experiment, and every variant, each
    setting's values alone in order then the changes together, the shortlist's marked with their
    label in the experiment and the measures they are among the best by."""
    labels = dict(zip(made.shortlist, (one.label for one in experiment.tried[1:]), strict=True))
    alone = [row for row in made.rows if len(row.values) == 1]
    current = next((row for row in made.rows if row.label == CURRENT), None)
    lines = [
        f"{made.id}: {len(made.rows) - 1} variants on {made.windows}, in {made.seconds / 60:.0f} min: "
        f"each setting's values alone from the current settings ({len(alone)}), then the best of "
        f"those changes together ({len(made.rows) - len(alone) - 1})."
        + (f" {len(made.refused)} could not be tried: {made.refused[0]}" if made.refused else ""),
        f"The shortlist is experiment {experiment.id}: the current settings and the best "
        f"{PER_MEASURE} at least by each measure, the image score, the fewest stripes, the windows "
        "passing and the depth of the weakest windows. None of them reads an image's quality: "
        f"judge by eye. Every variant (* in {experiment.id}, with the measures it is among the best "
        "by):",
    ]
    if current is not None:
        lines.append(f"- the current settings: {_scored(current.card)}")
    for row in [*alone, *(row for row in made.rows if len(row.values) > 1)]:
        among = made.shortlist.get(row.label)
        mark = f"* {experiment.id}:{labels[row.label]} ({', '.join(among)}) " if among else ""
        values = " + ".join(_said(name, value, row) for name, value in row.values.items())
        lines.append(f"- {mark}{row.label}: {values}: {_scored(row.card)}")
    return "\n".join(lines)


def _said(name: str, value: str, row: Row) -> str:
    """A setting's value in a variant, the segments' seconds with their crossings."""
    if name == "segments":
        return f"segments {row.changes['slicing']['segment_duration']:g} s ({value})"
    return f"{name} {value}"


class _Trying:
    """A search's variants as they are tried: their rows by label, and why some could not be."""

    def __init__(
        self,
        study: trials.Study,
        moved: Sequence[Axis],
        base: Mapping[str, Any],
        user: Given,
        profile: Profile,
        line: trials.LineState | None,
        windows: str,
        settings: Settings,
        config: QCConfig,
    ) -> None:
        self.study, self.moved, self.base, self.user = study, moved, base, user
        self.profile, self.line, self.settings, self.config = profile, line, settings, config
        self.parts = trials.parse_windows(windows)
        self.reach = trials.survey_reach(study)
        self.last = trials.last_xmids(study)
        self.rows: dict[str, Row] = {}
        self.refused: list[str] = []
        self._seen: set[str] = set()

    def run(self, points: Iterable[Mapping[str, int]]) -> None:
        """`points` tried, all at once: those tried before, or making the same settings, skipped;
        the empty point is the current settings' row."""
        todo: list[tuple[dict[str, str], dict[str, Any], trials.Trial, list[Any]]] = []
        for point in points:
            changes = changes_of(self.moved, point, self.base, self.profile)
            try:
                check_line(changes, self.user)
                merged = deep_merge(dict(self.base), changes)
                trial = trials.concrete(merged, self.profile, self.config, self.reach)
                chosen = trials.trial_windows(trial, self.profile, self.parts, self.line, self.last)
            except ValueError as error:
                self.refused.append(str(error))
                continue
            made = json.dumps(
                [trial.preset.model_dump(mode="json"), trial.near_field_m], sort_keys=True
            )
            if made in self._seen:
                continue
            self._seen.add(made)
            values = {
                axis.name: axis.values[point[axis.name]].name
                for axis in self.moved
                if axis.name in point
            }
            todo.append((values, changes, trial, chosen))
        stopping.check()
        runs = trials.tried_all(
            self.study,
            [(trial, chosen) for _, _, trial, chosen in todo],
            self.profile,
            self.settings,
            self.config,
        )
        for (values, changes, trial, _), (run, _) in zip(todo, runs, strict=True):
            numbered = sum(1 for name in self.rows if name != CURRENT)
            label = f"s{numbered + 1}" if values else CURRENT
            self.rows[label] = Row(
                label=label,
                values=values,
                changes=changes,
                trial=run.key,
                length=trial.preset.masw.length,
                card=_card(run),
            )


def _point(moved: Sequence[Axis], row: Row) -> dict[str, int]:
    """`row`'s point: the index of each value it changes, by setting."""
    return {
        axis.name: next(
            i for i, value in enumerate(axis.values) if value.name == row.values[axis.name]
        )
        for axis in moved
        if axis.name in row.values
    }


def _card(run: trials.TrialRun) -> dict[str, Any]:
    card = trials.card_of(run)
    return {
        "covered": len(card.covered),
        "windows": len(card.windows),
        "image": card.image,
        "image_p10": card.image_p10,
        "stripes": card.stripes,
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
    ("depth of the weakest", lambda card: card["depth_p10_m"]),
)


def _shortlist(rows: Sequence[Row], by_length: bool = True) -> dict[str, list[str]]:
    """The best by each measure in turn, PER_MEASURE of each at least, then deeper while there is
    room, SHORTLIST at most: each row's label with the measures it is among the best by; by window
    length, or as found (`by_length` false: the best first)."""
    ranked: list[tuple[str, list[Row]]] = []
    for name, value in MEASURES:
        counted = [
            row
            for row in rows
            if value(row.card) is not None
            and (name == "windows passing" or _passing(row.card) >= PASSING)
        ]
        ranked.append((name, sorted(counted, key=lambda row: -(value(row.card) or 0.0))))
    chosen: dict[str, list[str]] = {}
    depth = 0
    while any(depth < len(counted) for _, counted in ranked):
        for name, counted in ranked:
            if depth >= len(counted):
                continue
            label = counted[depth].label
            if label in chosen:
                chosen[label].append(name)
            elif len(chosen) < SHORTLIST:
                chosen[label] = [name]
        if len(chosen) >= SHORTLIST and depth + 1 >= PER_MEASURE:
            break
        depth += 1
    if not by_length:
        return chosen
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
        f"{part('image_p10', '.2f')}), stripes {part('stripes', '.3f')}, depth "
        f"{part('depth_m', '.0f')} m, beyond reach {part('beyond', '.0%')}"
    )


def _spread(low: float, high: float, count: int) -> list[float]:
    """`count` distances from `low` to `high` m, evenly in their logarithm, rounded."""
    if high <= low:
        return [float(round(low))]
    ratio = (high / low) ** (1 / (count - 1))
    return [float(round(low * ratio**i)) for i in range(count)]


def _spacing(profile: Profile) -> float:
    xs = sorted(receiver.x for receiver in profile.receivers)
    gaps = [b - a for a, b in pairwise(xs) if b > a]
    return min(gaps) if gaps else 1.0


def _mutes(profile: Profile, cone: Cone | None, config: QCConfig) -> tuple[Value, ...]:
    """PACo's mute candidates from the survey's cone (without one: none and the standard): each
    preprocesses every record anew, so a few, not a range."""
    if cone is None:
        vmin, vmax = config.mute.standard
        stages: dict[str, Mapping[str, Any] | None] = {
            "none": None,
            "standard": {"method": "mute", "vmin": vmin, "vmax": vmax},
        }
    else:
        stages = {
            str(name): stage for name, stage in mute_candidates(cone, config.mute, profile).items()
        }
    return tuple(
        Value(name, None, {"muting": stage if stage is not None else {"method": "none"}})
        for name, stage in stages.items()
    )


def _interferometry(config: QCConfig) -> list[Axis]:
    """A passive line's interferometry: the segments by their crossings of the window, the FK
    selection's threshold and band, the whitening, the normalization."""
    bands = (None, (30, 3000), (80, 1500), (50, 1000), (100, 800), (150, 600))
    return [
        Axis(
            "segments",
            tuple(
                Value(f"{n} crossings", n, {SEGMENTS: n / config.signal.vg_min}) for n in CROSSINGS
            ),
            (("slicing", "segment_duration"), ("slicing", "segment_step")),
        ),
        Axis(
            "fk_threshold",
            tuple(
                Value(f"{t:g}", t, {"selection": {"method": "fk", "threshold": t}})
                for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
            ),
            (("selection", "threshold"),),
        ),
        Axis(
            "fk_band",
            tuple(
                Value(
                    "open" if band is None else f"{band[0]}-{band[1]} m/s",
                    None,
                    {
                        "selection": {
                            "method": "fk",
                            "vmin": band[0] if band else None,
                            "vmax": band[1] if band else None,
                        }
                    },
                )
                for band in bands
            ),
            (("selection", "vmin"), ("selection", "vmax")),
        ),
        Axis(
            "whitening",
            tuple(
                Value(method, None, {"whitening": {"method": method}})
                for method in ("onebit", "onebit_apod", "none")
            ),
            (("whitening", "*"),),
        ),
        Axis(
            "normalization",
            tuple(
                Value(method, None, {"normalization": {"method": method}})
                for method in ("onebit", "none")
            ),
            (("normalization", "*"),),
        ),
    ]


def _stackings() -> tuple[Value, ...]:
    """The correlations' stacking: phase-weighted with nu 1 to 8, linear, root."""
    weighted = tuple(
        Value(f"phase_weighted nu {nu}", None, {"stacking": {"method": "phase_weighted", "nu": nu}})
        for nu in (1, 2, 3, 4, 6, 8)
    )
    return (
        *weighted,
        Value("linear", None, {"stacking": {"method": "linear"}}),
        Value("root 2", None, {"stacking": {"method": "root", "n": 2}}),
    )


def _given(axis: Axis, given: Given) -> bool:
    """Whether the user gave a setting `axis` sets: it stays as they gave it."""
    for stage, key in axis.sets:
        if stage == "near_field_m" or stage not in given.line:
            continue
        if key == "*" or key in given.line[stage]:
            return True
    return False


def _narrowed(axis: Axis, wanted: Sequence[Any] | None) -> Axis:
    """`axis` with the values `wanted` keeps: [low, high] for a number, names for a choice."""
    if wanted is None:
        return axis
    numbers = [w for w in wanted if isinstance(w, int | float) and not isinstance(w, bool)]
    if axis.values[0].number is not None and len(wanted) == 2 and len(numbers) == 2:
        low, high = sorted(float(w) for w in numbers)
        kept = tuple(v for v in axis.values if v.number is not None and low <= v.number <= high)
    else:
        names = {str(w) for w in wanted}
        kept = tuple(v for v in axis.values if v.name in names)
    if not kept:
        raise ValueError(
            f"ranges {axis.name}: none of its values, {', '.join(v.name for v in axis.values)}."
        )
    return Axis(axis.name, kept, axis.sets)
