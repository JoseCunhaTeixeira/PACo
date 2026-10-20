"""search: the full grid of the line's settings, every combination of the values the physics sets,
scored in code by the grid as a trial scores it; the shortlist, the best by several measures, an
experiment; once a study."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pytest
import yaml
from mcp import Client
from mcp.types import CallToolResult
from sigpipe.masw.profiles import load_profile
from sigpipe.masw.quality.cone import Cone

from paco.autonomous import server
from paco.autonomous.tools import grid, trials
from paco.autonomous.tools import search as searching
from paco.autonomous.tools.given import Given, give
from paco.autonomous.tools.search import Row
from paco.autonomous.tools.trials import new_study
from paco.call import shown
from paco.qc.config import QCConfig
from paco.qc.loops import deep_merge
from paco.settings import Settings

CONE = Cone(vmin=150.0, vmax=900.0, median=300.0, n_traces=40)
# The tests' own values, over grid.yaml's (which the user edits).
LENGTHS: dict[str, Any] = {"values": [5, 7, 10, 48, 96], "longest": 0.5}
PINNED: dict[str, Any] = {
    "group_velocities": {"slowest": {"value": 80}, "fastest": {"value": 1500}},
    "active": {
        "length": LENGTHS,
        "distance_max": {"values": [0.125, 0.25, 0.5, 1.0]},
        "near_field_m": {"values": [0.0, 0.5, 1.0]},
    },
    "passive-active": {"length": LENGTHS},
    "passive": {
        "length": LENGTHS,
        "segments": {"values": [2, 3, 4, 6, 8, 11, 16]},
        "fk_threshold": {"values": [round(0.1 * n, 1) for n in range(11)]},
        "fk_band": {"values": ["open", "group_velocities"]},
    },
}
GRID = searching.grid_of(PINNED)


def _names(axes: list[searching.Axis]) -> list[str]:
    return [axis.name for axis in axes]


def test_each_method_searches_the_settings_of_its_page(input_dir: Path) -> None:
    settings = Settings(input_dir=input_dir)
    active = load_profile("active_test_profile", settings)
    passive = load_profile("passive_test_profile", settings)

    shots = searching.axes(active, "active", 63.0, CONE, Given(), GRID)
    correlated = searching.axes(active, "passive-active", 63.0, CONE, Given(), GRID)
    noise = searching.axes(passive, "passive", None, None, Given(), GRID)

    # Every setting of PAC's page for the method, in its order, as grid.yaml gives them.
    assert _names(shots) == [
        "length",
        "distance_min",
        "distance_max",
        "near_field_m",
        "trigger",
        "muting",
        "filtering",
        "dispersion",
        "image_stacking",
    ]
    assert _names(correlated) == [
        "length",
        "distance_min",
        "distance_max",
        "near_field_m",
        "trigger",
        "muting",
        "filtering",
        "stacking",
        "dispersion",
    ]
    assert _names(noise) == [
        "length",
        "filtering",
        "segments",
        "fk_threshold",
        "fk_band",
        "whitening",
        "normalization",
        "stacking",
        "dispersion",
    ]
    axis = {a.name: a for a in shots}
    # The lengths up to half the line, the farthest shot as shares of the reach.
    assert [v.number for v in axis["length"].values] == [5, 7, 10, 48]
    assert [v.number for v in axis["distance_max"].values] == [7.9, 15.8, 31.5, 63.0]
    # The mutes from the cone and the grid's velocities, as PACo's mute_candidates makes them.
    made = {v.name: v.changes["muting"] for v in axis["muting"].values}
    assert made["none"] == {"method": "none"}
    assert (made["standard"]["vmin"], made["standard"]["vmax"]) == (80.0, 1500.0)
    assert (made["wider"]["vmin"], made["wider"]["vmax"]) == (round(150 / 2.25, 1), 2025.0)
    assert (made["tight"]["vmin"], made["tight"]["vmax"]) == (200.0, 450.0)
    # The FK thresholds 0 to 1, the band between the grid's velocities.
    noisy = {a.name: a for a in noise}
    assert [v.number for v in noisy["fk_threshold"].values] == [
        round(0.1 * n, 1) for n in range(11)
    ]
    assert noisy["fk_band"].values[1].changes["selection"] | {} == {
        "method": "fk",
        "vmin": 80.0,
        "vmax": 1500.0,
    }


def test_a_line_with_shots_takes_a_near_field_in_either_mode(input_dir: Path) -> None:
    settings = Settings(input_dir=input_dir)
    active = load_profile("active_test_profile", settings)
    passive = load_profile("passive_test_profile", settings)
    config = QCConfig()
    base = {"mode": "passive-active", "masw": {"length": 24}}

    far = trials.concrete(base, active, config, None)
    near = trials.concrete({**base, "near_field_m": 9.0}, active, config, None)

    def shots(trial: trials.Trial) -> int:
        return sum(len(window.selected_files) for window in trials.line_windows(active, trial))

    assert near.near_field_m == 9.0
    assert shots(near) < shots(far)
    with pytest.raises(ValueError, match="near_field_m: the modes' whose windows stack shots"):
        trials.concrete({"near_field_m": 3.0}, passive, config, None)


def test_a_study_changes_the_grid_and_never_its_file(input_dir: Path) -> None:
    profile = load_profile("passive_test_profile", Settings(input_dir=input_dir))
    written = (Path(searching.__file__).parent.parent / searching.GRID_FILE).read_text()
    changes = {
        "group_velocities": {"slowest": {"value": 160}},
        "passive": {"length": {"values": [5, 10]}},
    }

    changed = searching.grid_of(
        {**PINNED, **changes, "passive": PINNED["passive"] | changes["passive"]}
    )
    found = {a.name: a for a in searching.axes(profile, "passive", None, None, Given(), changed)}

    assert [v.number for v in found["length"].values] == [5, 10]
    # The segments' crossings at the slowest group velocity given: 2 crossings of 160 m/s.
    assert found["segments"].values[0].changes[searching.SEGMENTS] == 2 / 160
    file_lengths = yaml.safe_load(written)["passive"]["length"]["values"]
    assert list(searching.grid_of().passive["length"].values) == file_lengths
    assert (Path(searching.__file__).parent.parent / searching.GRID_FILE).read_text() == written
    with pytest.raises(ValueError, match="no setting colour"):
        searching.grid_of({"passive": {"colour": {"description": "", "unit": "", "values": [1]}}})


def test_the_grids_size_from_its_file(input_dir: Path) -> None:
    profile = load_profile("active_test_profile", Settings(input_dir=input_dir))
    grid_file = searching.grid_of(
        {
            **PINNED,
            "active": {**PINNED["active"], "length": {"values": [5, 48, 96], "longest": 0.5}},
        }
    )

    every, bases = searching.size("active", grid_file=grid_file)
    lined, lined_bases = searching.size("active", profile, grid_file)

    # 3 lengths x 4 distances x 3 near fields x 5 mutes x 2 stackings; 3 x 5 bases.
    assert (every, bases) == (360, 15)
    # On its 96 receivers, half the line at most: 96 left out, and no near field reaching the
    # farthest shot.
    assert lined_bases == 10
    assert lined < 240


def test_the_users_settings_stay_out_of_the_grid(input_dir: Path) -> None:
    profile = load_profile("active_test_profile", Settings(input_dir=input_dir))
    given = Given(line={"masw": {"length": 24}, "muting": {"method": "none"}})

    found = searching.axes(profile, "active", 63.0, CONE, given, GRID)

    assert "length" not in _names(found) and "muting" not in _names(found)


def test_every_combination_and_the_segments_follow_the_window(input_dir: Path) -> None:
    profile = load_profile("passive_test_profile", Settings(input_dir=input_dir))
    axes = searching.axes(profile, "passive", None, None, Given(), GRID)
    base = {"slicing": {"segment_duration": 1.0, "segment_step": 1.0}}
    segments = next(axis for axis in axes if axis.name == "segments")
    sixteen = next(i for i, value in enumerate(segments.values) if value.number == 16)
    five, ten = (
        next(i for i, value in enumerate(axes[0].values) if value.number == n) for n in (5, 10)
    )

    found = searching.points(axes)

    sizes = [len(axis.values) for axis in axes]
    assert len(found) == eval("*".join(map(str, sizes)))
    assert all(set(point) == set(_names(axes)) for point in found)
    # Short segments, 2 to 16 times the slowest wave (80 m/s) takes to cross the window, √2
    # apart, end to end: 16 crossings of 5 receivers (6 m) 1.2 s, of 10 (13.5 m) 2.7 s.
    assert [value.number for value in segments.values] == [2, 3, 4, 6, 8, 11, 16]
    for length, seconds in ((five, 1.2), (ten, 2.7)):
        made = searching.changes_of(axes, {"length": length, "segments": sixteen}, base, profile)
        assert made is not None
        assert made["slicing"] == {"segment_duration": seconds, "segment_step": seconds}


def test_the_farthest_shot_and_the_near_field_follow_the_window(input_dir: Path) -> None:
    profile = load_profile("active_test_profile", Settings(input_dir=input_dir))
    axes = searching.axes(profile, "active", 63.0, CONE, Given(), GRID)
    lengths = [int(value.number or 0) for value in axes[0].values]
    longest, seven = len(lengths) - 1, lengths.index(7)
    near = next(axis for axis in axes if axis.name == "near_field_m")
    half, whole = (i for i, value in enumerate(near.values) if value.number in (0.5, 1.0))

    floored = searching.changes_of(axes, {"length": longest, "distance_max": 0}, {}, profile)
    shared = searching.changes_of(
        axes, {"length": seven, "distance_max": 2, "near_field_m": half}, {}, profile
    )
    reaching = searching.changes_of(
        axes, {"length": longest, "distance_max": 0, "near_field_m": whole}, {}, profile
    )

    spacing = 1.5
    # No farthest shot nearer than the window's half length and two receivers.
    assert floored is not None
    assert floored["masw"]["distance_max"] == round((48 - 1) * spacing / 2 + 2 * spacing, 1)
    # The near field half the window: 7 receivers, 9 m, 4.5 m.
    assert shared is not None
    assert shared[trials.NEAR_FIELD] == 4.5
    # A near field reaching the farthest shot means nothing: no variant.
    assert reaching is None


def _row(label: str, length: int, values: dict[str, str], **card: float) -> Row:
    filled: dict[str, Any] = {
        "covered": 10,
        "windows": 12,
        "image": 0.75,
        "image_p10": 0.7,
        "stripes": 0.02,
        "band_hz": 40.0,
        "depth_m": 25.0,
        "depth_p10_m": 12.0,
        "beyond": 0.5,
        **card,
    }
    return Row(label=label, values=values, changes={}, trial=label, length=length, card=filled)


def test_the_shortlist_takes_the_best_by_each_measure_none_alike() -> None:
    def at(length: int, stacking: str = "nu 2") -> dict[str, str]:
        return {"length": str(length), "stacking": stacking}

    rows = [
        _row("g1", 14, at(14), image=0.9),
        # As g1 in its windows' settings, its stacking alone differs: left out.
        _row("g2", 14, at(14, "nu 4"), image=0.89),
        _row("g3", 5, at(5), stripes=0.008),
        _row("g4", 6, at(6), covered=3, stripes=0.001),
        _row("g5", 7, at(7), covered=12),
        _row("g6", 8, at(8), band_hz=80.0),
        _row("g7", 20, at(20), depth_p10_m=40.0),
        *(_row(f"g{n}", 9, at(9, f"nu {n}")) for n in range(8, 20)),
    ]

    found = searching._shortlist(rows)  # pyright: ignore[reportPrivateUsage]

    assert len(found) <= searching.SHORTLIST
    assert "image score" in found["g1"]
    assert "g2" not in found
    assert "fewest stripes" in found["g3"]
    assert "windows passing" in found["g5"]
    assert "widest band along the line" in found["g6"]
    assert "depth of the weakest" in found["g7"]
    # Fewest stripes with a quarter of its windows passing: a smear, not among the fewest stripes.
    assert "fewest stripes" not in found.get("g4", [])
    lengths = [next(row.length for row in rows if row.label == label) for label in found]
    assert lengths == sorted(lengths)


def test_the_table_pages_its_rows_sorted_and_filtered() -> None:
    rows = tuple(
        _row(
            f"g{n}", 5 + n % 3, {"length": str(5 + n % 3), "stacking": "nu 2"}, image=0.5 + n / 100
        )
        for n in range(40)
    )
    made = searching.Search(
        id="S1",
        windows="spread:2",
        axes={"length": ["5", "6", "7"], "stacking": ["nu 2"]},
        units={"length": "receivers", "stacking": "stage"},
        changes={},
        rows=rows,
        refused=0,
        refusal=None,
        shortlist={},
        experiment="E1",
        seconds=1.0,
        made_at=datetime.now(UTC),
    )

    first = searching.table(made)
    sevens = searching.table(made, where={"length": ["7"]}, sort="widest band along the line")

    assert "40 of its 40 variants, best first by image score, page 1 of 2" in first
    # The best image first; a setting with one value left out of each row.
    assert first.splitlines()[1].startswith("- g39: length 5: ")
    assert "stacking" not in first.splitlines()[1]
    assert sevens.startswith("S1: 13 of its 40 variants")
    assert all(line.startswith("- g") and "length 7" in line for line in sevens.splitlines()[1:])
    with pytest.raises(ValueError, match="sort: one of"):
        searching.table(made, sort="colour")
    with pytest.raises(ValueError, match="where length: no value 9"):
        searching.table(made, where={"length": ["9"]})
    with pytest.raises(ValueError, match="page: 1 to 2"):
        searching.table(made, page=3)


@pytest.mark.usefixtures("paco_env")
def test_settings_are_tried_once_the_grid_has_run() -> None:
    async def call() -> CallToolResult:
        async with Client(server.server) as client:
            return await client.call_tool(
                "try_settings",
                {"profile": "active_test_profile", "variants": [{"picking": {"fmax": 90.0}}]},
            )

    refused = anyio.run(call)

    assert refused.is_error and shown(refused).startswith(
        "Error executing tool try_settings: [precondition] Run search first: the full grid"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("profile", "base", "changes"),
    [
        (
            "active_test_profile",
            {"masw": {"length": 24}, "muting": {"method": "mute", "vmin": 80.0, "vmax": 1500.0}},
            [
                {"masw": {"length": 24, "distance_max": 30.0}},
                {
                    "masw": {"length": 24, "distance_max": 50.0},
                    "near_field_m": 9.0,
                    "image_stacking": {"method": "root", "n": 2},
                },
            ],
        ),
        (
            "active_test_profile",
            {"mode": "passive-active", "masw": {"length": 24}},
            [
                {"masw": {"length": 24, "distance_max": 30.0}},
                {"stacking": {"method": "phase_weighted", "nu": 4}},
            ],
        ),
        (
            "passive_test_profile",
            {"masw": {"length": 12}, "slicing": {"segment_duration": 1.0, "segment_step": 1.0}},
            [
                {"selection": {"method": "fk", "threshold": 0.1}},
                {
                    "selection": {"method": "fk", "threshold": 0.05, "vmin": 80, "vmax": 1500},
                    "stacking": {"method": "phase_weighted", "nu": 4},
                },
            ],
        ),
    ],
)
def test_the_grid_scores_a_variant_as_a_trial_does(
    input_dir: Path,
    tmp_path: Path,
    profile: str,
    base: dict[str, Any],
    changes: list[dict[str, Any]],
) -> None:
    settings = Settings(input_dir=input_dir, output_dir=tmp_path / "output", workers=2)
    config = QCConfig()
    study = new_study(profile, settings)
    line = load_profile(profile, settings)
    variants = {
        f"v{i}": trials.concrete(deep_merge(base, one), line, config, 63.0)
        for i, one in enumerate(changes)
    }

    runs, refused = grid.score(study, variants, "spread:2", None, line, settings, config)
    windows = trials.parse_windows("spread:2")
    tried = trials.tried_all(
        study,
        [
            (trial, trials.trial_windows(trial, line, windows, None, ()))
            for trial in variants.values()
        ],
        line,
        settings,
        config,
    )

    assert not refused
    assert not (study.folder / grid.GRID).exists()
    for run, (other, _) in zip(runs.values(), tried, strict=True):
        assert run.key == other.key
        assert [w.model_dump(exclude={"cached"}) for w in run.windows] == [
            w.model_dump(exclude={"cached"}) for w in other.windows
        ]


@pytest.mark.slow
def test_a_search_runs_the_grid_once_and_gives_its_shortlist(
    input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(input_dir=input_dir, output_dir=tmp_path / "output", workers=2)
    study = new_study("active_test_profile", settings)
    # The user's settings keep the grid small: the farthest shot and the near field alone.
    give(
        study,
        Given(
            line={
                "masw": {"length": 24},
                "muting": {"method": "none"},
                "image_stacking": {"method": "linear"},
            }
        ),
    )

    made, experiment = searching.search(
        study, line=None, settings=settings, config=QCConfig(), windows="spread:2"
    )
    said = searching.describe(made, experiment)

    moved = {name for name, values in made.axes.items() if len(values) > 1}
    assert moved == {"distance_max", "near_field_m"}
    assert made.axes["dispersion"] == ["fmin 0.0 fmax 100.0 vmin 1.0 vmax 1000.0 nv 1000"]
    assert {name for row in made.rows for name in row.values} <= set(made.axes)
    assert set(made.shortlist) <= {row.label for row in made.rows}
    assert len(experiment.tried) == len(made.shortlist) + 1
    saved = json.loads((study.folder / searching.SEARCHES / f"{made.id}.json").read_text())
    assert saved["experiment"] == experiment.id
    assert searching.searched(study) == made
    assert said.startswith(f"{made.id}: the full grid, {len(made.rows)} variants")
    assert f"The shortlist is experiment {experiment.id}" in said
    assert "- distance_max: " in said
    with pytest.raises(ValueError, match="The grid has run"):
        searching.search(study, line=None, settings=settings, config=QCConfig(), windows="spread:2")
