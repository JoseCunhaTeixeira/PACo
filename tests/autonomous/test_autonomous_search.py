"""search: the line's settings tried on a grid in code, each value alone from the current settings,
then the best of those changes together; the shortlist, the best by several measures, an
experiment; every variant in the table."""

import json
from pathlib import Path
from typing import Any

import pytest
from sigpipe.masw.profiles import load_profile
from sigpipe.masw.quality.cone import Cone

from paco.autonomous.tools import search as searching
from paco.autonomous.tools.given import Given
from paco.autonomous.tools.search import Row
from paco.autonomous.tools.trials import new_study
from paco.qc.config import QCConfig
from paco.settings import Settings

CONE = Cone(vmin=150.0, vmax=900.0, median=300.0, n_traces=40)


def _names(axes: list[searching.Axis]) -> list[str]:
    return [axis.name for axis in axes]


def test_each_method_searches_the_settings_of_its_page(input_dir: Path) -> None:
    settings = Settings(input_dir=input_dir)
    active = load_profile("active_test_profile", settings)
    passive = load_profile("passive_test_profile", settings)
    config = QCConfig()

    shots = searching.axes(active, "active", 63.0, CONE, config, Given())
    correlated = searching.axes(active, "passive-active", 63.0, CONE, config, Given())
    noise = searching.axes(passive, "passive", None, None, config, Given())

    assert _names(shots) == ["length", "distance_max", "muting", "near_field_m", "image_stacking"]
    assert _names(correlated) == ["length", "distance_max", "muting", "stacking"]
    assert _names(noise) == [
        "length",
        "segments",
        "fk_threshold",
        "fk_band",
        "whitening",
        "normalization",
        "stacking",
    ]
    # The lengths over the whole range, to half the line; the farthest shot from 10 m to the
    # line's length, the reach among them; PACo's mute candidates from the cone.
    lengths = [value.number for value in shots[0].values]
    assert lengths[0] == 5 and lengths[-1] == min(len(active.receivers) // 2, 48)
    distances = [value.number for value in shots[1].values]
    assert distances[0] == 10.0 and 63.0 in distances
    assert [value.name for value in shots[2].values] == [
        "none",
        "standard",
        "wider",
        "cone",
        "tight",
    ]


def test_the_users_settings_stay_and_ranges_narrow_the_rest(input_dir: Path) -> None:
    profile = load_profile("active_test_profile", Settings(input_dir=input_dir))
    given = Given(line={"masw": {"length": 24}, "muting": {"method": "none"}})

    found = searching.axes(
        profile,
        "active",
        63.0,
        CONE,
        QCConfig(),
        given,
        {"distance_max": [20, 40], "image_stacking": ["root 2"]},
    )

    assert _names(found) == ["distance_max", "near_field_m", "image_stacking"]
    assert all(20 <= (value.number or 0) <= 40 for value in found[0].values)
    assert [value.name for value in found[2].values] == ["root 2"]
    with pytest.raises(ValueError, match="no setting stacking on this line"):
        searching.axes(profile, "active", 63.0, CONE, QCConfig(), Given(), {"stacking": ["linear"]})
    with pytest.raises(ValueError, match="ranges distance_max: none of its values"):
        searching.axes(profile, "active", 63.0, CONE, QCConfig(), Given(), {"distance_max": [1, 2]})


def test_the_grid_tries_each_value_alone_and_the_segments_follow_the_window(
    input_dir: Path,
) -> None:
    profile = load_profile("passive_test_profile", Settings(input_dir=input_dir))
    axes = searching.axes(profile, "passive", None, None, QCConfig(), Given())
    base = {"masw": {"length": 5}, "slicing": {"segment_duration": 1.0, "segment_step": 1.0}}

    points = searching.grid(axes)
    segments = next(axis for axis in axes if axis.name == "segments")
    sixteen = next(i for i, value in enumerate(segments.values) if value.number == 16)
    nine = next(i for i, value in enumerate(axes[0].values) if value.number == 9)

    assert len(points) == sum(len(axis.values) for axis in axes)
    assert all(len(point) == 1 for point in points)
    # Short segments, 2 to 16 times the slowest wave (80 m/s) takes to cross the window, √2
    # apart, end to end: 16 crossings of 5 receivers (6 m) 1.2 s.
    assert [value.number for value in segments.values] == list(searching.CROSSINGS)
    assert searching.changes_of(axes, {"segments": sixteen}, base, profile) == {
        "slicing": {"segment_duration": 1.2, "segment_step": 1.2}
    }
    # With the length, the crossings of the variant's own window: 9 receivers, 12 m.
    made = searching.changes_of(axes, {"segments": sixteen, "length": nine}, base, profile)
    assert made["slicing"] == {"segment_duration": 2.4, "segment_step": 2.4}


def test_no_farthest_shot_nearer_than_the_windows_half_length(input_dir: Path) -> None:
    profile = load_profile("active_test_profile", Settings(input_dir=input_dir))
    axes = searching.axes(profile, "active", 63.0, CONE, QCConfig(), Given())
    length = next(axis for axis in axes if axis.name == "length")
    longest = len(length.values) - 1

    made = searching.changes_of(axes, {"length": longest, "distance_max": 0}, {}, profile)

    spacing = 1.5
    receivers = int(length.values[longest].number or 0)
    assert made["masw"]["distance_max"] == round((receivers - 1) * spacing / 2 + 2 * spacing, 1)


def _row(label: str, length: int, values: dict[str, str], **card: float) -> Row:
    filled: dict[str, Any] = {
        "covered": 10,
        "windows": 12,
        "image": 0.75,
        "image_p10": 0.7,
        "stripes": 0.02,
        "depth_m": 25.0,
        "depth_p10_m": 12.0,
        "beyond": 0.5,
        **card,
    }
    return Row(label=label, values=values, changes={}, trial=label, length=length, card=filled)


def test_the_shortlist_takes_the_best_by_each_measure() -> None:
    rows = [
        _row("s1", 14, {"length": "14"}, image=0.9),
        _row("s2", 5, {"stacking": "phase_weighted nu 4"}, stripes=0.008),
        _row("s3", 5, {"fk_threshold": "0.05"}, covered=3, stripes=0.001),
        _row("s4", 7, {"length": "7"}, covered=12),
        _row("s5", 20, {"length": "20"}, depth_p10_m=40.0),
        *(_row(f"s{n}", 5, {"overlap": f"{n}%"}) for n in range(6, 20)),
    ]

    found = searching._shortlist(rows)  # pyright: ignore[reportPrivateUsage]

    assert len(found) <= searching.SHORTLIST
    assert "image score" in found["s1"]
    assert "fewest stripes" in found["s2"]
    assert "windows passing" in found["s4"]
    assert "depth of the weakest" in found["s5"]
    # Fewest stripes with a quarter of its windows passing: a smear, left out.
    assert "s3" not in found
    lengths = [next(row.length for row in rows if row.label == label) for label in found]
    assert lengths == sorted(lengths)


def test_the_best_changes_are_tried_together(input_dir: Path) -> None:
    profile = load_profile("passive_test_profile", Settings(input_dir=input_dir))
    axes = searching.axes(profile, "passive", None, None, QCConfig(), Given())
    current = _row("current", 5, {})
    rows = [
        _row("s1", 5, {"normalization": "none"}, stripes=0.015),
        _row("s2", 5, {"stacking": "phase_weighted nu 4"}, stripes=0.012),
        _row("s3", 5, {"whitening": "onebit_apod"}, image=0.78),
        _row("s4", 5, {"whitening": "none"}, covered=3, image=0.9),
        _row("s5", 12, {"length": "12"}, image=0.85),
    ]

    points = searching.combined(rows, axes, current)

    named = [
        {name: next(a for a in axes if a.name == name).values[i].name for name, i in point.items()}
        for point in points
    ]
    # The three changes that beat the current settings without losing windows, each pair, all
    # three, and all three at the length the shortlist holds; never the whitening that lost
    # windows, nor the length on its own.
    assert {"normalization": "none", "stacking": "phase_weighted nu 4"} in named
    together = {
        "normalization": "none",
        "stacking": "phase_weighted nu 4",
        "whitening": "onebit_apod",
    }
    assert together in named
    assert {**together, "length": "12"} in named
    assert all(point.get("whitening") != "none" for point in named)
    assert len(named) == 5


@pytest.mark.slow
def test_a_search_tries_the_grid_and_gives_its_shortlist(
    input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(input_dir=input_dir, output_dir=tmp_path / "output", workers=2)
    study = new_study("active_test_profile", settings)
    ranges = {
        "length": [24, 32],
        "distance_max": [50, 80],
        "muting": ["none", "standard"],
        "near_field_m": ["none"],
        "image_stacking": ["linear"],
    }

    made, experiment = searching.search(
        study,
        current={"masw": {"length": 24}},
        line=None,
        settings=settings,
        config=QCConfig(),
        windows="spread:2",
        ranges=ranges,
    )
    said = searching.describe(made, experiment)

    assert made.rows[0].label == searching.CURRENT
    assert {name for row in made.rows[1:] for name in row.values} <= set(ranges)
    assert set(made.shortlist) <= {row.label for row in made.rows}
    assert len(experiment.tried) == len(made.shortlist) + 1
    saved = json.loads((study.folder / searching.SEARCHES / f"{made.id}.json").read_text())
    assert saved["experiment"] == experiment.id
    assert said.startswith(f"{made.id}: ")
    assert f"The shortlist is experiment {experiment.id}" in said
    assert "- the current settings: " in said
