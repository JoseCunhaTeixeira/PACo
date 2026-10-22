"""search: the full grid of the line's settings, every combination of the values the physics sets,
scored in code by the grid as a trial scores it; the really bad dropped, the others judged by eye
in sheets, the 3 cleanest of each mixed again until 3 are left, the cleanest of them chosen;
nothing tried after it; once a study."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
import yaml
from mcp import Client
from mcp.types import CallToolResult, ImageContent
from sigpipe.masw.profiles import load_profile
from sigpipe.masw.quality.cone import Cone

from paco.autonomous import server
from paco.autonomous.host import budgets
from paco.autonomous.tools import grid, look, trials
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
MUTES: dict[str, Any] = {"values": ["none", "standard", "wider", "cone", "tight"]}
STACKINGS: dict[str, Any] = {
    "values": [
        {"method": "linear"},
        {"method": "phase_weighted", "nu": 2},
        {"method": "phase_weighted", "nu": 4},
    ]
}
PINNED: dict[str, Any] = {
    "group_velocities": {"slowest": {"value": 80}, "fastest": {"value": 1500}},
    "active": {
        "length": LENGTHS,
        "distance_max": {"values": [0.125, 0.25, 0.5, 1.0]},
        "near_field_m": {"values": [0.0, 0.5, 1.0]},
        "muting": MUTES,
        "image_stacking": {"values": [{"method": "linear"}, {"method": "root", "n": 2}]},
    },
    "passive-active": {"length": LENGTHS, "muting": MUTES, "stacking": STACKINGS},
    "passive": {
        "length": LENGTHS,
        "segments": {"values": [2, 3, 4, 6, 8, 11, 16]},
        "fk_threshold": {"values": [round(0.1 * n, 1) for n in range(11)]},
        "fk_band": {"values": ["open", "group_velocities"]},
        "stacking": STACKINGS,
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


def _row(label: str, length: int, values: dict[str, str], **card: float | None) -> Row:
    filled: dict[str, Any] = {
        "covered": 10,
        "windows": 12,
        "empty": 0,
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


def test_the_really_bad_are_dropped_and_the_rest_sorted_into_sheets_by_length() -> None:
    def at(length: int, mute: str) -> dict[str, str]:
        return {"length": str(length), "muting": mute}

    fives = [_row(f"g{n}", 5, at(5, f"m{n:02d}")) for n in range(30)]
    sevens = [_row(f"g{n}", 7, at(7, f"m{n:02d}")) for n in range(30, 43)]

    sheets = searching.first_sheets(list(reversed([*fives, *sevens])), ["length", "muting"])

    # No image anywhere, or fewer than a quarter of the windows passing: dropped.
    assert searching.really_bad(_row("a", 5, {}, image=None).card)
    assert searching.really_bad(_row("b", 5, {}, covered=2).card)
    assert not searching.really_bad(_row("c", 5, {}, covered=3).card)
    # Twelve a sheet, one length a sheet, by length then by the settings' values.
    assert [len(sheet.labels) for sheet in sheets] == [12, 12, 6, 12, 1]
    assert [sheet.number for sheet in sheets] == [1, 2, 3, 4, 5]
    assert sheets[0].labels[:2] == ("g0", "g1")
    assert set(sheets[3].labels) | set(sheets[4].labels) == {row.label for row in sevens}


def _study_with_sheets(folder: Path, rows: list[Row]) -> trials.Study:
    study = trials.Study("active_test_profile", folder)
    (folder / searching.SEARCHES).mkdir(parents=True)
    made = searching.Search(
        id="S1",
        windows="spread:12",
        axes={"length": sorted({str(row.length) for row in rows}), "muting": ["a", "b"]},
        units={"length": "receivers", "muting": "name"},
        changes={},
        rows=tuple(rows),
        refused=0,
        refusal=None,
        places=(15.0, 50.0, 90.0, 125.0),
        seconds=60.0,
        made_at=datetime.now(UTC),
    )
    (folder / searching.SEARCHES / "S1.json").write_text(made.model_dump_json())
    knockout = searching.Knockout(
        round=1, sheets=searching.first_sheets(rows, ["length", "muting"])
    )
    (folder / searching.SEARCHES / searching.KNOCKOUT).write_text(knockout.model_dump_json())
    return study


def test_the_knockout_keeps_the_cleanest_round_after_round(tmp_path: Path) -> None:
    rows = [
        _row(f"g{n}", 5 if n < 20 else 9, {"length": "5" if n < 20 else "9", "muting": "a"})
        for n in range(40)
    ]
    study = _study_with_sheets(tmp_path / "study", rows)

    waiting = searching.unjudged(study)
    with pytest.raises(ValueError, match="Sheet 9 is not one left in round 1"):
        searching.judge(study, {9: ["g0", "g1", "g2"]}, "why")
    with pytest.raises(ValueError, match="Sheet 1: keep 3 of its variants"):
        searching.judge(study, {1: ["g0", "g1"]}, "why")
    with pytest.raises(ValueError, match="Sheet 1: no variant g30 on it"):
        searching.judge(study, {1: ["g30", "g0", "g1"]}, "why")
    # Round 1: 20 variants of 5 receivers in 2 sheets, 20 of 9 in 2; 3 kept of each.
    first = searching.judge(study, {1: ["g3", "g4", "g5"], 2: ["g12", "g15", "g19"]}, "one ridge")
    second = searching.judge(study, {3: ["g20", "g21", "g31"], 4: ["g39", "g32", "g33"]}, "fewer")
    # Round 2: the 12 kept, one sheet; 3 kept, the finalists.
    finalists = searching.judge(study, {1: ["g15", "g31", "g3"]}, "the smoothest")
    waiting_final = searching.unjudged(study)
    # The final, the agent's own: one of the finalists.
    with pytest.raises(ValueError, match="g30 is not a finalist: g15, g31, g3"):
        searching.choose(study, "g30", "why")
    final = searching.choose(study, "g31", "unbroken at every place")
    judged = [json.loads(line) for line in (study.folder / "searches" / searching.JUDGED).open()]

    assert (
        waiting == "Every sheet is judged first: 4 of round 1's 4 left; search again judges them."
    )
    assert (first.round, [sheet.number for sheet in first.left]) == (1, [3, 4])
    assert second.round == 2 and len(second.sheets) == 1 and not second.final
    assert second.sheets[0].labels == (
        "g3", "g4", "g5", "g12", "g15", "g19", "g20", "g21", "g31", "g39", "g32", "g33",
    )  # fmt: skip
    assert finalists.final and finalists.finalists == ("g15", "g31", "g3")
    assert waiting_final is not None and waiting_final.startswith(
        "Choose first the cleanest of the search's 3 finalists (choose)"
    )
    assert final.best == "g31" and final.done and searching.chosen(study) == "g31"
    assert searching.unjudged(study) is None
    # Each sheet's judgment recorded: what it showed, what was kept, why.
    assert [(one["round"], one["sheet"], one["kept"]) for one in judged] == [
        (1, 1, ["g3", "g4", "g5"]), (1, 2, ["g12", "g15", "g19"]),
        (1, 3, ["g20", "g21", "g31"]), (1, 4, ["g39", "g32", "g33"]),
        (2, 1, ["g15", "g31", "g3"]), (3, 1, ["g31"]),
    ]  # fmt: skip
    assert judged[0]["shown"] == [f"g{n}" for n in range(12)] and judged[0]["why"]
    with pytest.raises(ValueError, match=r"the search chose g31\. Commit it"):
        searching.judge(study, {1: ["g15"]}, "why")
    with pytest.raises(ValueError, match="The search chose g31 already"):
        searching.choose(study, "g15", "why")


def test_the_kept_are_mixed_into_new_sheets_until_three_are_left() -> None:
    survivors = [f"g{n}" for n in range(15)]

    mixed = searching.next_round(2, survivors)
    final = searching.next_round(3, ["g4", "g7", "g9"])
    alone = searching.next_round(3, ["g4"])

    # More than a sheet: dealt in turn into the fewest sheets, each holding the kept of different
    # sheets before.
    assert [sheet.labels for sheet in mixed.sheets] == [
        tuple(survivors[0::2]),
        tuple(survivors[1::2]),
    ]
    assert not mixed.finalists
    # Three left: the finalists, their sheet the final; one left: the choice already.
    assert final.final and final.finalists == ("g4", "g7", "g9") and final.best is None
    assert alone.done and alone.best == "g4" and not alone.left


def test_a_sheet_is_drawn_from_the_variants_thumbnails(tmp_path: Path) -> None:
    def thumb(xmid: float) -> grid.Thumb:
        image = np.tile(np.linspace(0, 255, grid.THUMB, dtype=np.uint8)[:, None], grid.THUMB)
        pick = (np.array([10.0, 20.0], dtype=np.float32), np.array([300.0, 250.0], np.float32))
        return grid.Thumb(xmid, image, (0.0, 100.0, 1.0, 1000.0), 0.33, pick, (3.0, 54.0))

    grid.write_thumbs(tmp_path / "g1.npz", [thumb(15.0), thumb(50.0), thumb(90.0), thumb(125.0)])
    grid.write_thumbs(tmp_path / "g2.npz", [thumb(15.0), thumb(50.0)])

    read = grid.read_thumbs(tmp_path / "g2.npz")
    drawn = look.draw_sheet(tmp_path, ["g1", "g2"], {"g1": 7, "g2": 13}, (15.0, 50.0, 90.0, 125.0))

    assert [one.xmid for one in read] == [15.0, 50.0]
    assert read[0].extent == (0.0, 100.0, 1.0, 1000.0) and read[0].lines_m == (3.0, 54.0)
    assert np.array_equal(read[0].image, thumb(15.0).image)
    assert np.array_equal(read[1].pick[1], np.array([300.0, 250.0], np.float32))
    assert drawn.startswith(b"\x89PNG")


def test_judging_the_sheets_spends_no_model_calls_and_no_cpu(tmp_path: Path) -> None:
    study = trials.Study("active_test_profile", tmp_path)
    at = datetime.now(UTC)
    for use in (
        budgets.Use(kind="model", phase="P2", at=at, seconds=2.0),
        budgets.Use(kind="tool", phase="P2", at=at, seconds=10.0, name="commit"),
        budgets.Use(kind="model", phase="P2", at=at, seconds=2.0),
        budgets.Use(kind="tool", phase="P2", at=at, seconds=300.0, name="search"),
        budgets.Use(
            kind="form", phase="P2", at=at, seconds=2.0, name="judge",
            prompt_tokens=9_000, completion_tokens=300,
        ),
        budgets.Use(kind="tool", phase="P2", at=at, seconds=5.0, name="judge"),
        budgets.Use(
            kind="form", phase="P2", at=at, seconds=2.0, name="judge",
            prompt_tokens=8_000, completion_tokens=200,
        ),
        budgets.Use(kind="tool", phase="P2", at=at, seconds=20.0, name="judge"),
    ):  # fmt: skip
        budgets.record(study, use)

    spent = budgets.spent(study, "quick")

    assert spent.model_calls == 2
    assert spent.cpu == 1.0
    # The judge's forms count in the tokens, which bound the run.
    assert spent.tokens == 17_500 and not spent.tokens_spent
    assert "0.0M of 2.0M tokens" in spent.said()


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
def test_a_search_runs_the_grid_once_and_sorts_it_into_sheets(
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

    made, knockout = searching.search(
        study, line=None, settings=settings, config=QCConfig(), windows="spread:6"
    )
    said = searching.describe(made, knockout)

    moved = {name for name, values in made.axes.items() if len(values) > 1}
    assert moved == {"distance_max", "near_field_m"}
    assert made.axes["dispersion"] == ["fmin 0.0 fmax 100.0 vmin 1.0 vmax 1000.0 nv 1000"]
    assert {name for row in made.rows for name in row.values} <= set(made.axes)
    assert searching.searched(study) == made
    assert knockout is not None and knockout == searching.knockout_of(study)
    shown = [label for sheet in knockout.sheets for label in sheet.labels]
    bad = {row.label for row in made.rows if searching.really_bad(row.card)}
    assert made.dropped == len(bad) and not bad & set(shown)
    assert len(shown) + len(bad) == len(made.rows)
    # Each variant shown kept at its windows nearest the places along the line, as PAC draws it.
    images = study.folder / searching.SEARCHES / searching.IMAGES
    assert len(made.places) == grid.PLACES
    for label in shown:
        thumbs = grid.read_thumbs(images / f"{label}.npz")
        assert 1 <= len(thumbs) <= grid.PLACES
        assert [one.xmid for one in thumbs] == sorted(one.xmid for one in thumbs)
        assert all(one.image.shape == (grid.THUMB, grid.THUMB) for one in thumbs)
    assert not any((images / f"{label}.npz").exists() for label in bad)
    assert said.startswith(f"{made.id}: the full grid, {len(made.rows)} variants")
    assert "dropped as really bad" in said and "image " not in said
    assert searching.unjudged(study) is not None
    with pytest.raises(ValueError, match="The grid has run"):
        searching.search(study, line=None, settings=settings, config=QCConfig(), windows="spread:6")


@pytest.mark.slow
def test_every_sheet_is_judged_until_the_search_chooses(paco_env: Settings) -> None:
    profile = "active_test_profile"
    study = new_study(profile, paco_env)
    # Windows of 21 receivers: more of the grid's variants give images than a final holds.
    give(
        study,
        Given(
            line={
                "masw": {"length": 21},
                "muting": {"method": "none"},
                "image_stacking": {"method": "linear"},
            }
        ),
    )

    async def calls() -> list[CallToolResult]:
        async with Client(server.server) as client:
            made = [
                await client.call_tool(
                    "search", {"profile": profile, "windows": "spread:6"}, meta={"vision": True}
                ),
                await client.call_tool(
                    "commit",
                    {"profile": profile, "changes": {}, "reasons": {}},
                    meta={"vision": True},
                ),
            ]
            # Each sheet shown judged, its first variants kept, as many as it asks, until the final.
            while (knockout := searching.knockout_of(study)) is not None and not (
                knockout.done or knockout.final
            ):
                kept = {
                    str(one.number): list(one.labels[: searching.to_keep(knockout, one)])
                    for one in knockout.left[:3]
                }
                made.append(
                    await client.call_tool(
                        "judge",
                        {"profile": profile, "kept": kept, "why": "the first of each"},
                        meta={"vision": True},
                    )
                )
            # The final, the agent's: its first finalist chosen.
            final = searching.knockout_of(study)
            assert final is not None and final.final
            made.append(
                await client.call_tool(
                    "choose",
                    {"profile": profile, "label": final.finalists[0], "why": "the first"},
                )
            )
            made.append(
                await client.call_tool(
                    "try_settings",
                    {"profile": profile, "variants": [{"picking": {"fmax": 90.0}}]},
                )
            )
            return made

    searched, refused, *judged, chose, tried = anyio.run(calls)

    done = searching.knockout_of(study)
    records = [
        json.loads(line)
        for line in (study.folder / searching.SEARCHES / searching.JUDGED).read_text().splitlines()
    ]
    first_round = sum(1 for one in records if one["round"] == 1)
    # The grid's summary in a text of its own, then its first sheets, each a figure.
    assert shown(searched).startswith("S1: the full grid")
    assert searched.content[1].text.startswith("Round 1, sheet 1 of ")  # pyright: ignore[reportAttributeAccessIssue]
    assert [type(one) for one in searched.content[2:]] == [ImageContent] * min(3, first_round)
    # Nothing committed before the search chose.
    assert refused.is_error and "[precondition] Every sheet is judged first" in shown(refused)
    # Every sheet judged and recorded; the final's sheet shown for the agent, then its choice,
    # recorded too, said with its changes; nothing tried after it.
    assert done is not None and done.best is not None and done.experiment is None
    assert done.best == done.finalists[0] and len(done.finalists) <= searching.KEEP
    assert all(one["why"] for one in records) and records[-1]["kept"] == [done.best]
    if judged:
        assert "The final: the " in shown(judged[-1])
        assert any(isinstance(one, ImageContent) for one in judged[-1].content)
    assert shown(chose).startswith(f"The search's choice: {done.best}")
    assert searching.unjudged(study) is None
    assert tried.is_error and "the search saw every combination by eye" in shown(tried).lower()
