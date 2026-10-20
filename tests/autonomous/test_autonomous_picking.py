"""pick, decide and freeze (docs/autonomous_plan.md §4.6, §4.7, §8): a scripted agent picks
active_test_profile's windows, decides every one and freezes its curves, through the autonomous agent's tools;
decide refuses a blocking flag, an unjustified note and a rejection before the window was tried
hard; a commit picks each window again with its recipe; nothing is processed or picked after the
freeze. On active_test_profile in 24-receiver windows (73 windows): the play takes some 40 s."""

import hashlib
import importlib.util
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pytest
from autonomous_judged import JudgedLine
from mcp import Client
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import find_run

from paco.autonomous import server
from paco.autonomous.sensors.flags import FlagClass
from paco.autonomous.tools.decide import owed_flags
from paco.autonomous.tools.pick import How, Picked, Recipe, check_recipe, picks_of
from paco.autonomous.tools.trials import Study
from paco.autonomous.tools.windows import select
from paco.call import shown
from paco.qc.log import latest, read_attempts
from paco.qc.replay import replay_run
from paco.settings import Settings, get_settings

L1 = {"masw": {"length": 24, "distance_max": 30.0}, "near_field_m": 8.25}
L1_REASONS = {
    "masw length": "a middle length between the survey's limits",
    "masw distance_max": "the shots near each window, for a quick test",
    "near_field_m": "half the longest wavelength PACo's trials reached",
}
# G3's own suggestion for the curves too narrow.
FIXES = {"min_relative_coherence": 0.3}
# A ridge's band, around the line's surface waves.
LASSO = [[10, 230], [10, 420], [40, 400], [40, 230]]
# The air wave, a corridor of 2 % around 338 m/s and loose rules: a pick G3 blocks.
AIR_WAVE = {
    "guide": [[15, 338], [90, 338]],
    "picking": {
        "corridor": 0.02,
        "fmin": 15,
        "fmax": 90,
        "min_relative_coherence": 0.1,
        "point_min_ratio": 0.5,
        "mode_min_ratio": 0.5,
    },
}
REJECTED = (
    "too_few_points, narrow_span, no_ridge, on_data, constant_wavelength, mode_jump and "
    "prominence remain after G3's fixes and a guide from the neighbours"
)
# The play: each call, and whether the tool refuses it.


def ways(*asked: dict[str, Any]) -> dict[str, Any]:
    """A pick call's arguments: each way of picking with its windows."""
    return {"ways": list(asked)}


PLAY: list[tuple[str, dict[str, Any], bool]] = [
    ("survey", {}, False),
    ("commit", {"changes": L1, "reasons": L1_REASONS}, False),
    ("pick", ways({"windows": "verdict:retry", "how": "auto", "picking": FIXES}), False),
    # Some of those windows have no passing neighbour to follow: said, the others picked.
    ("pick", ways({"windows": "verdict:retry", "how": "guide", "guide": "neighbours"}), False),
    # Two ways in one call.
    (
        "pick",
        ways(
            {"windows": "xmids:[29.25, 30.75]", "how": "lasso", "polygon": LASSO},
            {"windows": "xmids:[125.25]", "how": "delete"},
        ),
        False,
    ),
    (
        "commit",
        {"changes": {"picking": {"fmax": 90}}, "reasons": {"picking fmax": "the band G2 keeps"}},
        False,
    ),
    ("pick", ways({"windows": "xmids:[66.75]", "how": "guide", **AIR_WAVE}), False),
    ("decide", {"windows": "xmids:[66.75]", "status": "accepted", "reason": "fine"}, True),
    ("pick", ways({"windows": "xmids:[66.75]", "how": "auto"}), False),
    ("decide", {"windows": "verdict:pass", "status": "accepted"}, True),
    (
        "decide",
        {"windows": "verdict:pass", "status": "accepted", "reason": "notes read with care"},
        False,
    ),
    ("decide", {"windows": "status:pending", "status": "rejected", "reason": "no ridge"}, True),
    ("pick", ways({"windows": "xmids:[125.25]", "how": "guide", "guide": "neighbours"}), False),
    # A window accepted, rejected only once it was tried hard: its neighbours' guide first.
    ("decide", {"windows": "xmids:[24.75]", "status": "rejected", "reason": "no ridge"}, True),
    ("pick", ways({"windows": "status:pending", "how": "guide", "guide": "neighbours"}), False),
    # A window in two ways of one call: refused.
    (
        "pick",
        ways({"windows": "xmids:[17.25]", "how": "auto"}, {"windows": "xmids:17", "how": "auto"}),
        True,
    ),
    (
        "decide",
        {"windows": "status:pending+verdict:pass", "status": "accepted", "reason": "near field"},
        False,
    ),
    ("decide", {"windows": "status:pending", "status": "rejected", "reason": REJECTED}, False),
    ("freeze", {}, False),
    ("pick", ways({"windows": "all", "how": "auto"}), True),
    # After the freeze, decide is the models': none yet to accept.
    ("decide", {"windows": "all", "status": "accepted", "reason": "fine"}, True),
    ("commit", {"changes": {"picking": {"fmax": 80}}, "reasons": {"picking": "late"}}, True),
]


@dataclass(frozen=True)
class Played:
    """The play: each call's answer, whether it was refused, and the study it made."""

    answers: list[str]
    refused: list[bool]
    settings: Settings
    study: Study
    run: Path


@pytest.fixture(scope="module")
def played(
    input_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
    searched: None,  # noqa: ARG001
) -> Iterator[Played]:
    root = tmp_path_factory.mktemp("picking")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.setenv("PACO_INPUT_DIR", str(input_dir))
        patch.setenv("PACO_OUTPUT_DIR", str(root / "output"))
        patch.setenv("PACO_WORKERS", "4")
        get_settings.cache_clear()

        async def play() -> list[tuple[bool, str]]:
            async with Client(server.server) as client:
                found: list[tuple[bool, str]] = []
                for name, arguments, _ in PLAY:
                    result = await client.call_tool(
                        name, {"profile": "active_test_profile", **arguments}
                    )
                    found.append((bool(result.is_error), shown(result)))
                return found

        results = anyio.run(play)
        settings = get_settings()
        (folder,) = (settings.output_dir / "active_test_profile" / "studies").iterdir()
        line = json.loads((folder / "line.json").read_text())
        yield Played(
            [text for _, text in results],
            [refused for refused, _ in results],
            settings,
            Study("active_test_profile", folder),
            find_run(line["run_id"], settings),
        )
    get_settings.cache_clear()


def answer(played: Played, step: int) -> str:
    return played.answers[step]


@pytest.mark.slow
def test_the_scripted_agent_decides_every_window_and_freezes(played: Played) -> None:
    # Each call answered, or refused where the play expects it.
    assert played.refused == [refused for _, _, refused in PLAY], played.answers
    frozen = json.loads((played.study.folder / "curves.json").read_text())
    assert len(frozen["curves"]) + len(frozen["rejected"]) == 73
    assert [one["unit"] for one in frozen["rejected"]][:2] == ["xmid_18.75", "xmid_26.25"]
    assert answer(played, 18).startswith(
        f"Frozen, from L2 (run {played.run.name}): 51 curves accepted of the line's 73 windows, "
        "22 rejected."
    )


@pytest.mark.slow
def test_decide_refuses_a_blocking_flag(played: Played) -> None:
    assert answer(played, 6).splitlines()[1].startswith("- 66.75: reject;")
    assert "air_wave (blocking)" in answer(played, 6)
    assert answer(played, 7).endswith(
        "- 66.75: blocking flag sharpness, air_wave: it cannot be accepted until a later pick "
        "clears it"
    )


@pytest.mark.slow
def test_decide_refuses_an_unjustified_note(played: Played) -> None:
    said = answer(played, 9)

    assert "accepting curves with notes needs a reason, one line that covers them" in said
    assert "near_field at xmid " in said
    assert (
        answer(played, 10).splitlines()[1].startswith("The reason covers their notes: near_field")
    )


@pytest.mark.slow
def test_decide_refuses_a_rejection_before_the_window_was_tried_hard(played: Played) -> None:
    rows = answer(played, 11).splitlines()[1:]

    # The windows whose line pick carried fixable flags were picked again (the play's third call):
    # they owe no more, whatever those picks raised; 125.25 still owes its neighbours' guide.
    assert rows == [
        "- 125.25: not tried hard yet: owes a pick guided by its neighbours (how=guide, "
        "guide=neighbours): 2 on one side pass G3"
    ]
    # An accepted window: its rejection owes the same tries.
    assert answer(played, 13).endswith(
        "- 24.75: not tried hard yet: owes a pick guided by its neighbours (how=guide, "
        "guide=neighbours): 2 on one side pass G3"
    )
    # Rejected once the reason names the flags none of its picks fixed; its picks listed.
    assert answer(played, 17).splitlines()[1] == (
        "- 18.75 had: line (retry), auto (retry), guide (neighbours) (retry), guide (neighbours) "
        "(retry), guide (neighbours) (retry)."
    )


def test_a_rejection_owes_a_pick_for_the_line_picks_fixable_flags_alone() -> None:
    def picked(how: How, flags: dict[str, FlagClass] | None = None) -> Picked:
        return Picked(
            run_id="r",
            unit="xmid_18.75",
            recipe=Recipe(how=how),
            flags=flags or {},
            at=datetime.now(UTC),
        )

    line = picked(
        "line", {"too_few_points": "fixable", "narrow_span": "fixable", "near_field": "note"}
    )
    later = picked("auto", {"no_ridge": "fixable", "on_data": "fixable"})

    # The line's pick's fixable flags, a reason naming one in words settling it.
    assert owed_flags([line], "") == ["too_few_points", "narrow_span"]
    assert owed_flags([line], "Its narrow span stays: the ridge is short") == ["too_few_points"]
    # A pick of the model's after it settles them, its own new flags owing nothing; a deletion is
    # no attempt; a commit's new line pick owes its own.
    assert owed_flags([line, later], "") == []
    assert owed_flags([line, picked("delete")], "") == ["too_few_points", "narrow_span"]
    assert owed_flags([line, later, picked("line", {"on_data": "fixable"})], "") == ["on_data"]


@pytest.mark.slow
def test_a_commit_picks_each_window_again_with_its_recipe(played: Played) -> None:
    assert answer(played, 5).startswith(
        f"L2 changed the line's settings in run {played.run.name}. The windows' own recipes, "
        "picked again: 37 (guide 28, lasso 2, auto 6, delete 1)."
    )
    again = [one for one in picks_of(played.study, played.run.name) if one.reapplied]
    assert len({one.unit for one in again}) == 37
    assert {"xmid_29.25", "xmid_30.75", "xmid_125.25"} <= {one.unit for one in again}


@pytest.mark.slow
def test_a_lasso_guides_the_tracker_inside_its_band(played: Played) -> None:
    attempt = latest(read_attempts(played.run), "xmid_29.25", "picking")
    # The lasso's pick, as the study keeps it: the window was picked again after it.
    lasso = next(
        one
        for one in picks_of(played.study, played.run.name)
        if one.unit == "xmid_29.25" and one.recipe.how == "lasso"
    )

    assert attempt is not None
    assert attempt.triggered_by == "backtrack"
    assert lasso.parameters is not None
    parameters = PickingParameters.model_validate(lasso.parameters)
    assert (parameters.fmin, parameters.fmax) == (10, 40)
    assert parameters.guide is not None
    assert all(10 <= f <= 40 and 230 <= v <= 420 for f, v in parameters.guide)
    # The ridge it holds stands out too little: G3 blocks it.
    assert "- 29.25: reject; prominence (blocking)" in answer(played, 4)


@pytest.mark.slow
def test_several_ways_of_picking_in_one_call(played: Played) -> None:
    lines = answer(played, 4).splitlines()

    # Each way's windows under it, G4 once after them.
    assert lines[0] == "Picked 3 windows in 2 ways:"
    assert lines[1] == "On a lasso of 4 vertices:" and lines[4:6] == [
        "Their curves deleted:",
        "- 125.25: its curve deleted",
    ]
    assert lines[6].startswith("G4 on the line now:") and len(lines) == 7
    # A window in two ways of one call: refused, nothing picked.
    assert answer(played, 15).endswith("xmid 17.25 is in two ways: a window in one of them.")


@pytest.mark.slow
def test_the_frozen_curves_hold_what_the_inversion_needs(played: Played) -> None:
    frozen = json.loads((played.study.folder / "curves.json").read_text())

    assert (frozen["run_id"], frozen["commit"], frozen["length"]) == (played.run.name, 2, 24)
    assert (frozen["dx_m"], frozen["length_m"], frozen["reach_m"]) == (1.5, 34.5, 66.64)
    assert frozen["settings"] == {**L1, "picking": {"fmax": 90}}
    for curve in frozen["curves"]:
        path = played.run / curve["curve"]
        assert curve["curve"] == f"{curve['unit']}/{CURVES_FILE}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == curve["sha256"]
        assert curve["points"] >= 2
    assert frozen["rejected"][:2] == [
        {"unit": "xmid_18.75", "xmid": 18.75, "reason": REJECTED},
        {"unit": "xmid_26.25", "xmid": 26.25, "reason": REJECTED},
    ]


@pytest.mark.slow
def test_nothing_is_processed_or_picked_after_the_freeze(played: Played) -> None:
    for step in (19, 21):
        assert "[precondition] The curves are frozen" in answer(played, step)
    # decide goes on, on the models of the frozen curves (18.75 was rejected before): none is
    # there to accept before an inversion.
    assert answer(played, 20).startswith(
        "Error executing tool decide: [bad argument] Nothing decided, 51 of the models cannot be "
        "accepted:\n- 17.25: it has no model: invert it, or reject it\n"
    )


@pytest.mark.slow
def test_paco_replay_makes_the_picked_line_again(played: Played) -> None:
    replay = replay_run(played.run.name, played.settings)

    assert replay.same
    # The deleted window's curve was picked again (step 12); the others too: every window's curve.
    curves = [one for one in replay.compared if one.stage == "picking"]
    assert len(curves) == 73


@pytest.mark.slow
def test_the_windows_a_spec_names(judged_line: JudgedLine) -> None:
    def named(spec: str, statuses: dict[str, str] | None = None) -> list[str]:
        return [
            unit.removeprefix("xmid_") for unit in select(judged_line.folder, spec, statuses or {})
        ]

    assert named("all") == ["17.25", "53.25", "89.25", "125.25"]
    assert named("xmids:40-100") == ["53.25", "89.25"]
    assert named("xmids:[50, 120]") == ["53.25", "125.25"]
    assert named("flag:near_field+xmids:80-130") == ["89.25", "125.25"]
    assert named("status:accepted", {"xmid_53.25": "accepted"}) == ["53.25"]
    assert named("status:pending", {"xmid_53.25": "accepted"}) == ["17.25", "89.25", "125.25"]
    with pytest.raises(ValueError, match="No window at 300 m"):
        named("xmids:[300]")
    with pytest.raises(ValueError, match="No window where verdict:retry"):
        named("verdict:retry")
    with pytest.raises(ValueError, match="Unknown windows 'everything'"):
        named("everything")
    # One position needs no list; a + keeps what every part names, and says how to list them.
    assert named("xmids:89") == ["89.25"]
    with pytest.raises(ValueError, match=r"to name several windows, list them, as xmids:\["):
        named("xmids:10-60+89.25")


@pytest.mark.parametrize(
    ("recipe", "said"),
    [
        ({"how": "guide"}, "how=guide: give guide"),
        ({"how": "guide", "guide": [[10, 200]]}, "2 points"),
        ({"how": "lasso", "polygon": [[10, 200], [20, 200]]}, "3 vertices"),
        ({"how": "auto", "polygon": [[10, 200], [20, 200], [20, 300]]}, "polygon is for how=lasso"),
        ({"how": "auto", "guide": "neighbours"}, "guide is for how=guide"),
        ({"how": "auto", "picking": {"corridors": 0.1}}, "picking: corridors: Extra inputs"),
        ({"how": "auto", "picking": {"corridor": 2}}, "picking: corridor: Input should be less"),
    ],
)
def test_a_recipe_is_refused_saying_why(recipe: dict[str, Any], said: str) -> None:
    with pytest.raises(ValueError, match=said):
        check_recipe(Recipe.model_validate(recipe), PickingParameters())


@pytest.mark.slow
def test_the_soils_come_from_the_frozen_curves(played: Played) -> None:
    # Last: it writes the run's soil columns. With PACo's petro extra only.
    if not all(importlib.util.find_spec(name) for name in ("santiludo", "keras", "keras_nlp")):
        pytest.skip("needs PACo's petro extra")
    frozen = json.loads((played.study.folder / "curves.json").read_text())
    accepted = {curve["unit"] for curve in frozen["curves"]}

    async def calls() -> tuple[str, str]:
        async with Client(server.server) as client:
            asked = {"profile": "active_test_profile"}
            models = await client.call_tool("petro_models", asked)
            soils = await client.call_tool(
                "invert_petro", {**asked, "model": "grand_est_15-50hz_193-415mps"}
            )
            return shown(models), shown(soils)

    models, soils = anyio.run(calls)

    # The frozen curves alone counted, inverted and judged (G7, G8), PAC's sections written.
    assert f"of the {len(accepted)} curves G4 passed" in models
    assert soils.startswith(f"Soils of run {played.run.name}, from its frozen curves.")
    assert "G7: " in soils and "G8: " in soils
    inverted = {
        attempt.unit
        for attempt in read_attempts(played.run)
        if attempt.stage == "petro_inversion" and attempt.unit != "line"
    }
    assert inverted and inverted <= accepted
    assert list(played.run.glob("PetroInversion_Section_*.png"))
