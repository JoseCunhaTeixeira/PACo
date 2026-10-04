"""Before a stage redoes work already there, or changes work made by hand, the user chooses: the
tools say what is there and give the options, doing nothing, and the agent calls the option the
request says or asks; work made by hand is replaced only once the user could reply. The same
conversation goes on with its own work without asking."""

import json
import shutil
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, RequestParamsMeta, TextResourceContents
from sigpipe.base.dispersion_curve import DispersionCurve, Mode
from sigpipe.masw.picks import CURVES_FILE, load_curves, save_pick
from sigpipe.masw.runs import RunError, find_run, load_image, load_manifest
from sigpipe.masw.runs.origin import mark_edited

from paco import choices, inspection, server
from paco.agent.conversion import result_for_model
from paco.agent.scope import READ_ONLY as READ_ONLY_FOR_THE_HOST
from paco.qc import StageResult, run_work
from paco.qc.log import read_attempts
from paco.qc.origin import MODEL_FILES, SOIL_FILES
from paco.settings import Settings, get_settings

SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A short sampler: every step of an inversion, in about a second per window.
SHORT = {"n_iterations": 500, "n_burnin_iterations": 50, "n_chains": 2}
M0 = Mode("M", 0)


def _call(
    name: str,
    arguments: dict[str, Any],
    conversation: str,
    turn: int | None = None,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A tool's result, called in `conversation` (at `turn`, for a message of `scope`) as the
    host does: its structured content."""

    async def call() -> CallToolResult:
        async with Client(server.server) as client:
            meta: RequestParamsMeta = {"conversation": conversation}
            if turn is not None:
                meta["turn"] = turn
            if scope is not None:
                meta["scope"] = scope
            return await client.call_tool(name, arguments, meta=meta)

    result = anyio.run(call)
    assert not result.is_error, result.content
    content = result.structured_content
    assert content is not None
    return content.get("result", content)


def _scope(
    *asked: str, redo: bool = False, hand: str = "unsaid", chosen: str | None = None
) -> dict[str, Any]:
    """A message's scope as PACo's host sends it."""
    return {"asked": list(asked), "redo": redo, "hand_work": hand, "chosen": chosen}


def _hand_curve(window: Path) -> DispersionCurve:
    """An M0 picked by hand in PAC, in place of the window's."""
    image = load_image(window)
    fs = np.linspace(10.0, 40.0, 16)
    curve = DispersionCurve(fs=fs, vs=350.0 - 4.0 * fs, mode=M0, acquisition=image.acquisition)
    save_pick(window, image, curve)
    mark_edited(window, M0)
    return curve


@pytest.fixture(scope="module")
def picked(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """active_p1 processed and picked by the assistant in an earlier conversation."""
    root = tmp_path_factory.mktemp("choices")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PACO_INPUT_DIR", str(demo_input_dir))
        patch.setenv("PACO_OUTPUT_DIR", str(root / "outputs"))
        patch.setenv("PACO_WORKERS", "2")
        patch.chdir(root)
        server.get_settings.cache_clear()
        run_id = _call(
            "run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}, "earlier"
        )["run_id"]
        _call("pick", {"run_id": run_id}, "earlier")
    server.get_settings.cache_clear()
    return root / "outputs", run_id


@pytest.fixture
def run(
    picked: tuple[Path, str], demo_input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Settings, str, str]]:
    """A copy of the picked run, the server writing there, and a new conversation's id."""
    source, run_id = picked
    shutil.copytree(source, tmp_path / "outputs")
    monkeypatch.setenv("PACO_INPUT_DIR", str(demo_input_dir))
    monkeypatch.setenv("PACO_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PACO_WORKERS", "2")
    monkeypatch.chdir(tmp_path)
    server.get_settings.cache_clear()
    yield server.get_settings(), run_id, uuid.uuid4().hex
    server.get_settings.cache_clear()


def test_a_profile_with_a_run_is_processed_again_only_as_the_user_chooses(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run

    asked = _call("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}, new)

    assert asked["run_id"] == run_id
    assert asked["summary"].startswith(f"Run {run_id} (the assistant's, ")
    assert "4 images, " in asked["summary"]
    assert asked["next"].startswith(
        "Nothing was done. Asked to process, the user chooses how, unless their request says "
        "which option: ask them with these options"
    )
    # A new run, or the run there to work on: its curves, a client sending no scope.
    assert (
        '(1) a new run: run_processing(profile="active_p1", overrides={"masw": {"length": 24, '
        '"step": 24}}, again=true)' in asked["next"]
    )
    assert (
        f"(2) work on run {run_id} (active, windows of 24 receivers, curves): "
        f'pick(run_id="{run_id}")' in asked["next"]
    )
    assert len(list((settings.output_dir / "active_p1").iterdir())) == 1  # nothing done
    again = _call(
        "run_processing",
        {"profile": "active_p1", "overrides": SMALL_WINDOWS, "again": True},
        new,
    )
    assert again["run_id"] != run_id
    # The same conversation goes on with its own run: processed again, no question.
    goes_on = _call("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}, new)
    assert goes_on["run_id"] not in (run_id, again["run_id"])


def test_a_profile_asked_with_other_windows_still_offers_its_runs(
    run: tuple[Settings, str, str],
) -> None:
    # Other windows than the run there's: the user chooses still, the run said with its own.
    settings, run_id, new = run

    asked = _call(
        "run_processing", {"profile": "active_p1", "overrides": {"masw": {"length": 12}}}, new
    )

    assert (
        '(1) a new run: run_processing(profile="active_p1", overrides={"masw": {"length": 12}}, '
        "again=true)" in asked["next"]
    )
    assert f"(2) work on run {run_id} (active, windows of 24 receivers, curves)" in asked["next"]
    assert len(list((settings.output_dir / "active_p1").iterdir())) == 1


def test_the_run_chosen_goes_on_to_the_first_stage_asked(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run

    def go_on(*stages: str) -> str:
        asked = choices.Conversation(new, 1, choices.Asked(frozenset(stages)))
        planned = choices.processing("active_p1", asked, settings, {})
        assert planned is not None and planned.options[0][0] == "a new run"
        return planned.options[1][1]

    # The run has curves and no model: inverted; picking asked, picked; nothing after, read.
    assert go_on("process", "invert") == f'invert(run_id="{run_id}")'
    assert go_on("process", "pick", "invert") == f'pick(run_id="{run_id}")'
    assert go_on("process") == f'inspect(what="run", run_id="{run_id}")'


def test_a_stage_on_an_earlier_run_waits_for_the_choice_processing_gives(
    run: tuple[Settings, str, str],
) -> None:
    # Asked to process and pick, the model picks the run there first: the choice comes before.
    settings, run_id, new = run
    run_folder = find_run(run_id, settings)
    before = run_work(run_folder, load_manifest(run_id, settings))

    refused = _call("pick", {"run_id": run_id}, new, scope=_scope("process", "pick"))

    assert refused["status"] == "refused"
    assert refused["next"] == (
        'Nothing was done. Call run_processing(profile="active_p1") first: it gives the user\'s '
        "choice of a new run or a run to work on."
    )
    assert run_work(run_folder, load_manifest(run_id, settings)) == before


def test_the_workers_a_message_asks_run_its_work(
    run: tuple[Settings, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, new = run
    used: list[int] = []
    process_line = server.qc.process_line

    def recorded(profile: str, overrides: Any, settings: Settings, *more: Any) -> Any:  # noqa: ANN401
        used.append(settings.workers)
        return process_line(profile, overrides, settings, *more)

    monkeypatch.setattr(server.qc, "process_line", recorded)
    asked = {**_scope("process", redo=True), "workers": 3}

    processed = _call(
        "run_processing",
        {"profile": "active_p1", "overrides": SMALL_WINDOWS, "again": True},
        new,
        scope=asked,
    )

    # Three, not the settings' two; said with the parameters used.
    assert used == [3]
    assert processed["used"][-1] == "3 workers: as your message asked"


def test_more_workers_than_cores_run_on_the_cores_and_say_so() -> None:
    result = StageResult(run_id="r", summary="Processed.", next="", used=("mode active",))

    assert server.with_workers(result, 2, 12).used == (
        "mode active",
        "2 workers: as your message asked",
    )
    assert server.with_workers(result, 64, 12).used[-1] == (
        "12 workers: the machine's cores, fewer than the 64 asked"
    )
    assert server.with_workers(result, None, 12) is result


def test_the_host_reads_the_runs_from_the_servers_resources(
    run: tuple[Settings, str, str],
) -> None:
    _, run_id, _ = run

    async def read() -> list[Any]:
        async with Client(server.server) as client:
            texts: list[Any] = []
            for uri in ("paco://profiles/active_p1/runs", f"paco://runs/{run_id}", "paco://runs"):
                found = await client.read_resource(uri)
                content = found.contents[0]
                assert isinstance(content, TextResourceContents)
                texts.append(json.loads(content.text))
            with pytest.raises(MCPError):
                await client.read_resource("paco://runs/20990101-000000-abcd")
            return texts

    listed, described, newest = anyio.run(read)

    one = {
        "run_id": run_id,
        "profile": "active_p1",
        "label": "active, windows of 24 receivers, curves",
        "went": "curves",
    }
    assert listed == {"runs": [one]} == newest
    assert described == one


def test_a_run_with_curves_is_picked_as_the_user_chooses(run: tuple[Settings, str, str]) -> None:
    settings, run_id, new = run
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    before = run_work(run_folder, manifest)

    asked = _call("pick", {"run_id": run_id}, new)

    assert asked["next"].startswith("Nothing was done. Asked to pick, the user chooses how")
    assert f'pick(run_id="{run_id}", windows="all")' in asked["next"]
    assert f'pick(run_id="{run_id}", positions=["<m>"])' in asked["next"]
    assert run_work(run_folder, manifest) == before
    missing = [unit for unit, one in before.items() if one.image and one.m0 is None]
    if missing:
        assert f'pick(run_id="{run_id}", windows="missing")' in asked["next"]


def test_a_curve_picked_by_hand_is_replaced_only_once_the_user_could_reply(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run
    run_folder = find_run(run_id, settings)
    window = run_folder / "xmid_8.88"
    hand = _hand_curve(window)
    every = {"run_id": run_id, "windows": "all"}

    asked = _call("pick", every, new, turn=1)
    # Replaced in the turn the question was shown: the user has not replied, asked again.
    unanswered = _call("pick", {**every, "hand": "replace"}, new, turn=1)
    kept = _call("pick", {**every, "hand": "keep"}, new, turn=2)

    assert asked["summary"] == "Curves picked by hand at xmid 8.88 (1), which pick would change."
    assert "it is replaced only once they have replied" in asked["next"]
    assert 'windows="all", hand="replace")' in asked["next"]
    assert unanswered == asked
    assert "Retries" in kept["summary"]
    saved = load_curves(window)
    assert saved is not None
    np.testing.assert_allclose(saved.dispersion_curves[0].vs, hand.vs, rtol=1e-3)
    # The user kept it: replacing it later asks again.
    assert _call("pick", {**every, "hand": "replace"}, new, turn=3) == asked

    _call("pick", {**every, "hand": "replace"}, new, turn=4)

    saved = load_curves(window)
    assert saved is not None and not np.allclose(saved.dispersion_curves[0].vs[:3], hand.vs[:3])
    (aside,) = (window / "by_hand").iterdir()
    assert aside.name.startswith("picking_") and (aside / "DispersionCurves_0000.csv").exists()


def test_models_and_soil_columns_already_there_are_the_users_choice(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    # A model and a soil column made in PAC's inversion pages, at xmid 8.88.
    window = run_folder / "xmid_8.88"
    (window / MODEL_FILES.replace("*", "Model_0000_best.csv")).write_text("depth,vs\n")
    (window / SOIL_FILES.replace("*", "Model_0000.csv")).write_text("depth,soil\n")

    conversation = choices.Conversation(new)
    inverting = choices.inverting(run_folder, manifest, conversation, None, None, None)
    everything = choices.inverting(run_folder, manifest, conversation, None, "all", None)
    kept = choices.inverting(run_folder, manifest, conversation, None, "all", "keep")
    soils = choices.soils(run_folder, manifest, conversation, "a model", None, None)

    assert isinstance(inverting, choices.Choice)
    assert "1 model (1 by hand)" in inverting.summary
    assert isinstance(everything, choices.Choice)
    assert everything.summary.startswith("Models made by hand at xmid 8.88")
    assert isinstance(kept, choices.Plan) and kept.units is not None
    assert "xmid_8.88" not in kept.units and not kept.replace_hand
    assert isinstance(soils, choices.Choice)
    assert "1 soil column (1 by hand)" in soils.summary
    # Worked on in the conversation: its own work, gone on with.
    choices.worked(conversation, run_id)
    assert isinstance(
        choices.inverting(run_folder, manifest, conversation, None, None, None), choices.Plan
    )


def test_positions_narrow_every_window_or_those_without_one(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    conversation = choices.Conversation(new)
    at_9 = ["xmid_8.88"]

    missing = choices.inverting(run_folder, manifest, conversation, at_9, "missing", None)
    every = choices.inverting(run_folder, manifest, conversation, at_9, "all", None)

    assert missing == choices.Plan(units=at_9) and every == choices.Plan(units=at_9)
    with pytest.raises(RunError, match="has a curve in every window asked"):
        choices.picking(run_folder, manifest, conversation, at_9, "missing", None)


def test_every_window_of_a_run_without_curves_is_its_first_pick(
    run: tuple[Settings, str, str],
) -> None:
    # Picked as a first pick (not an asked one): the earlier stages G2 and G3 blame are done
    # again once, and no starting value reads as a change.
    settings, run_id, new = run
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    for window in manifest.windows:
        (run_folder / window.folder / CURVES_FILE).unlink(missing_ok=True)

    planned = choices.picking(run_folder, manifest, choices.Conversation(new), None, "all", None)

    assert planned == choices.Plan(units=None)


def test_a_redo_over_work_made_by_hand_asks_first(run: tuple[Settings, str, str]) -> None:
    settings, run_id, _ = run
    run_folder = find_run(run_id, settings)
    _hand_curve(run_folder / "xmid_8.88")
    work = run_work(run_folder, load_manifest(run_id, settings))

    unknown = choices.Conversation()

    asked = choices.redoing(
        unknown,
        run_id,
        "phase_shift",
        ["xmid_2.88", "xmid_8.88"],
        work,
        None,
        {"dispersion": {"vmax": 900}},
    )

    assert asked is not None and asked.summary.startswith("Work made by hand at xmid 8.88")
    assert '"changes": ' not in asked.next and 'changes={"dispersion": {"vmax": 900}}' in asked.next
    assert choices.redoing(unknown, run_id, "phase_shift", ["xmid_2.88"], work, None, None) is None
    assert (
        choices.redoing(unknown, run_id, "phase_shift", ["xmid_8.88"], work, "keep", None) is None
    )


def test_a_result_of_one_of_two_models_reaches_the_model_unwrapped() -> None:
    wrapped = CallToolResult(
        content=[], structured_content={"result": {"job_id": "j", "state": "queued"}}
    )

    assert json.loads(result_for_model(wrapped)) == {"job_id": "j", "state": "queued"}


# ---------------------------------------------------------------- with the message's scope


def test_a_stage_the_message_does_not_ask_goes_on_from_the_run(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run
    invert = _scope("invert")

    processed = _call("run_processing", {"profile": "active_p1"}, new, 1, invert)
    picked = _call("pick", {"run_id": run_id}, new, 1, invert)

    # No question: the way on, from the run's curves.
    for result in (processed, picked):
        assert result["next"].endswith(
            f'Go on from this run without asking: invert(run_id="{run_id}").'
        )
        assert result["options"] == []
    assert processed["next"].startswith(
        f"Nothing was done: this message does not ask to process active_p1, whose run {run_id} "
        "is there."
    )
    assert len(list((settings.output_dir / "active_p1").iterdir())) == 1


def test_work_there_is_asked_about_unless_the_message_asks_to_redo_it(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    before = run_work(run_folder, manifest)

    # Every window again, and a new run, unasked by the message: the question, nothing done.
    asked = _call("pick", {"run_id": run_id, "windows": "all"}, new, 1, _scope("pick"))
    again = _call(
        "run_processing", {"profile": "active_p1", "again": True}, new, 1, _scope("process")
    )

    assert asked["next"].startswith("Nothing was done. Asked to pick, the user chooses how")
    labels = [option["label"] for option in asked["options"]]
    assert "pick every window again" in labels
    assert again["next"].startswith("Nothing was done. Asked to process")
    assert run_work(run_folder, manifest) == before
    # Asked to redo: done, without a question.
    redone = _call("pick", {"run_id": run_id}, new, 1, _scope("pick", redo=True))
    assert "Retries" in redone["summary"] and redone["options"] == []


def test_hand_work_is_replaced_when_the_message_asks_it_or_chose_it(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run
    window = find_run(run_id, settings) / "xmid_8.88"
    hand = _hand_curve(window)
    every = {"run_id": run_id, "windows": "all"}

    # The model's own replace, the message silent on it: the question, in any turn.
    unasked = _call("pick", {**every, "hand": "replace"}, new, 5, _scope("pick", redo=True))
    assert unasked["summary"] == "Curves picked by hand at xmid 8.88 (1), which pick would change."
    assert [option["label"] for option in unasked["options"]] == [
        "keep it as it is",
        "replace it, theirs set aside in the window's by_hand folder",
    ]
    saved = load_curves(window)
    assert saved is not None
    np.testing.assert_allclose(saved.dispersion_curves[0].vs, hand.vs, rtol=1e-3)

    # The option chosen in the next message: replaced.
    chosen = unasked["options"][1]["call"]
    _call("pick", {**every, "hand": "replace"}, new, 6, _scope("pick", chosen=chosen))

    saved = load_curves(window)
    assert saved is not None and not np.allclose(saved.dispersion_curves[0].vs[:3], hand.vs[:3])
    # The message's own words, in its first turn: replaced too.
    _hand_curve(window)
    _call("pick", every, new, 7, _scope("pick", redo=True, hand="replace"))
    assert len(list((window / "by_hand").iterdir())) == 2


def test_what_a_message_asks_holds_for_its_own_stages_only(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)

    def conversation(**asked: Any) -> choices.Conversation:  # noqa: ANN401
        return choices.Conversation(new, 1, choices.Asked(**asked))

    invert_again = conversation(stages=frozenset({"invert"}), redo=True)
    # Picking unasked, at a position whose curve is there: no pick, the way on.
    at_9 = choices.picking(run_folder, manifest, invert_again, ["xmid_8.88"], None, None)
    # "Redo the inversion" asks no picking again: every window's curve stays.
    every = choices.picking(run_folder, manifest, invert_again, None, "all", None)

    for planned in (at_9, every):
        assert isinstance(planned, choices.Choice)
        assert planned.next.endswith(
            f'Go on from this run without asking: invert(run_id="{run_id}").'
        )
    assert invert_again.allows("invert", 'windows="all"')
    assert not invert_again.allows("pick", 'windows="all"')
    # Hand work replaced where the message asked it, and there only.
    curves_too = conversation(stages=frozenset({"pick"}), hand_work="replace")
    assert curves_too.hand(None, "pick", "pick") == "replace"
    assert curves_too.hand("keep", "pick", "pick") == "replace"  # the user's words first
    assert curves_too.hand(None, "redo", "process") is None
    assert curves_too.hand("replace", "invert", "invert") is None
    chose = conversation(stages=frozenset({"pick"}), chosen='pick(run_id="r", hand="replace")')
    assert chose.hand("replace", "pick", "pick") == "replace"
    assert chose.hand("replace", "invert_petro", "soils") is None
    # Keep or replace is the user's to say, not the model's: a call's keep waits for their
    # answer, unless they chose the offered call that keeps it.
    unsaid = conversation(stages=frozenset({"pick"}))
    assert unsaid.hand("keep", "pick", "pick") is None
    kept = conversation(
        stages=frozenset({"pick"}), chosen='pick(run_id="r", windows="all", hand="keep")'
    )
    assert kept.hand("keep", "pick", "pick") == "keep"


def test_processing_unasked_goes_on_to_the_first_stage_asked(
    run: tuple[Settings, str, str],
) -> None:
    # Asked to pick and invert a profile whose curves are there: picking comes first, and asks.
    settings, run_id, new = run
    asked = choices.Conversation(new, 1, choices.Asked(frozenset({"pick", "invert"})))

    planned = choices.processing("active_p1", asked, settings, {})

    assert planned is not None
    assert planned.next.endswith(f'Go on from this run without asking: pick(run_id="{run_id}").')


def _inverted(
    arguments: dict[str, Any], conversation: str, turn: int, scope: dict[str, Any] | None = None
) -> dict[str, Any]:
    """invert's job, with a short sampler, followed to its end: its last status."""
    status = _call("invert", {**arguments, "parameters": SHORT}, conversation, turn, scope)
    while status["state"] in ("queued", "running"):
        status = _call("job_status", {"job_id": status["job_id"]}, conversation, turn, scope)
    return status


def test_a_window_without_a_curve_among_those_asked_is_left_out_not_the_batch(
    run: tuple[Settings, str, str],
) -> None:
    # xmid 2.88's curve removed: an inversion asked there and at 9 m inverts xmid 8.88, and
    # says which window it left out.
    settings, run_id, new = run
    (find_run(run_id, settings) / "xmid_2.88" / CURVES_FILE).unlink()

    status = _inverted({"run_id": run_id, "positions": [3, 9]}, new, 1)

    assert (status["state"], status["total"]) == ("succeeded", 1)
    assert "Left out, without a curve the inversion takes: xmid 2.88 (1)." in status["notes"]


def test_with_a_scope_the_positions_are_the_messages(run: tuple[Settings, str, str]) -> None:
    _, run_id, new = run
    at_9 = {**_scope("invert"), "positions_m": [9.0]}

    # The model's own positions: the message's instead.
    status = _inverted({"run_id": run_id, "positions": [15, 21]}, new, 1, at_9)

    assert status["total"] == 1
    assert any(note.startswith("Positions: 9 m: xmid 8.88, window") for note in status["notes"])


def test_picking_changes_are_picking_settings() -> None:
    with pytest.raises(
        ValueError, match=r"Not picking settings: mode\. The picking settings: "
    ) as error:
        server.checked_picking({"mode": "M1", "threshold": 0.4})
    # What to do next: pick M0 without them, and where the higher modes are picked.
    assert "Call pick again without them: it picks M0." in str(error.value)
    assert server.checked_picking({"threshold": 0.4}) == {"threshold": 0.4}


def test_a_run_whose_manifest_does_not_read_is_not_gone_on_from(
    run: tuple[Settings, str, str],
) -> None:
    # A run being written, or broken, beside the profile's run: skipped, never the tool's
    # failure.
    settings, run_id, new = run
    broken = settings.output_dir / "active_p1" / "20260930-235959-ffff"
    broken.mkdir()
    (broken / "run.json").write_text("{}")

    asked = _call("run_processing", {"profile": "active_p1"}, new)
    runs = inspection.runs_text(settings)

    assert asked["run_id"] == run_id
    assert f"Run {run_id}: active_p1 (active)" in runs
    assert (
        "Run 20260930-235959-ffff: its manifest does not read (being written, or broken)." in runs
    )


# ---------------------------------------------------------------- what each tool does to a run


def test_the_tools_say_whether_they_only_read() -> None:
    async def listed() -> list[Any]:
        async with Client(server.server) as client:
            return list((await client.list_tools()).tools)

    tools = anyio.run(listed)

    read_only = {
        tool.name for tool in tools if tool.annotations and tool.annotations.read_only_hint
    }
    assert read_only == server.READ_ONLY == READ_ONLY_FOR_THE_HOST
    assert all(tool.annotations is not None for tool in tools)


def test_read_only_tools_change_nothing_and_a_run_keeps_its_calls(
    run: tuple[Settings, str, str],
) -> None:
    settings, run_id, new = run
    stamps = {path: path.stat().st_mtime_ns for path in settings.output_dir.rglob("*")}

    _call("inspect", {"what": "run", "run_id": run_id}, new, 1)
    _call("inspect", {"what": "window", "run_id": run_id, "position": 9}, new, 1)
    _call("petro_models", {"run_id": run_id}, new, 1)
    _call("preset_settings", {"profile": "active_p1"}, new, 1)

    assert {path: path.stat().st_mtime_ns for path in settings.output_dir.rglob("*")} == stamps
    # A call that changes the run leaves its trace there, with its conversation and turn.
    _call("judge", {"run_id": run_id}, new, 2, _scope("pick"))
    lines = (find_run(run_id, settings) / server.CALLS_FILE).read_text().splitlines()
    calls = [json.loads(line) for line in lines]
    # The fixture's own conversation processed and picked the run: its calls are there too.
    assert [call["tool"] for call in calls if call["conversation"] == "earlier"] == [
        "run_processing",
        "pick",
    ]
    (traced,) = [call for call in calls if call["conversation"] == new]
    assert (traced["conversation"], traced["turn"], traced["tool"]) == (new, 2, "judge")
    assert traced["arguments"] == {"run_id": run_id, "positions": None}
    assert traced["scope"]["asked"] == ["pick"] and traced["outcome"] in ("ok", "partial")


def test_a_pick_keeps_the_picking_values_the_user_gave(run: tuple[Settings, str, str]) -> None:
    settings, run_id, new = run
    before = datetime.now(UTC)
    every = {"run_id": run_id, "windows": "all", "changes": {"min_relative_coherence": 0.5}}

    picked = _call("pick", every, new, 1, _scope("pick", redo=True))

    assert picked["kept"] == ["min_relative_coherence 0.5"]
    attempts = [  # the windows' (the line's G4 attempt has no picking parameters)
        attempt
        for attempt in read_attempts(find_run(run_id, settings))
        if attempt.stage == "picking"
        and attempt.started_at >= before
        and attempt.unit.startswith("xmid_")
    ]
    assert attempts and all(
        attempt.parameters["min_relative_coherence"] == 0.5 for attempt in attempts
    )


@pytest.mark.usefixtures("paco_env")
def test_the_windows_the_message_gives_are_the_runs_whatever_the_model_wrote() -> None:
    # "Windows of 24 receivers, every 24": the model wrote them in metres, which on active_p1's
    # 0.25 m spacing would be 97 receivers; the message's words decide (its scope).
    scope = {**_scope("process"), "window": {"length": 24, "step": 24}}
    wrong = {"masw": {"length_m": 24, "step_m": 24}, "dispersion": {"vmax": 900}}

    result = _call(
        "run_processing", {"profile": "active_p1", "overrides": wrong}, "windows", 1, scope
    )

    manifest = load_manifest(result["run_id"], get_settings())
    assert (manifest.preset.masw.length, manifest.preset.masw.step) == (24, 24)
    # The model's other settings kept.
    assert manifest.preset.dispersion.vmax == 900  # pyright: ignore[reportAttributeAccessIssue]


def test_the_lengths_the_message_compares_are_the_variants_whatever_the_model_wrote() -> None:
    def asked(compared: dict[str, tuple[float, ...]]) -> choices.Conversation:
        scope = choices.Asked(frozenset({"process"}), compared=compared)
        return choices.Conversation(id="c", turn=1, asked=scope)

    # "Windows of 3 m and of 6 m": the model wrote receivers; each variant keeps its own
    # other settings.
    wrong = [
        {"masw": {"length": 3, "step": 1}, "dispersion": {"vmax": 900}},
        {"masw": {"length": 6, "step": 1}},
    ]
    assert server._compared_variants(asked({"length_m": (3.0, 6.0)}), wrong) == [  # pyright: ignore[reportPrivateUsage]
        {"masw": {"step": 1, "length_m": 3.0}, "dispersion": {"vmax": 900}},
        {"masw": {"step": 1, "length_m": 6.0}},
    ]
    # Four lengths over the model's two variants: each with what both share, receivers whole.
    four = server._compared_variants(  # pyright: ignore[reportPrivateUsage]
        asked({"length": (12.0, 24.0, 36.0, 48.0)}), wrong
    )
    assert four == [{"masw": {"step": 1, "length": n}} for n in (12, 24, 36, 48)]
    # More than compare takes: said, nothing compared.
    with pytest.raises(ValueError, match="4 variants at most"):
        server._compared_variants(asked({"length": (5.0, 7.0, 9.0, 11.0, 13.0)}), wrong)  # pyright: ignore[reportPrivateUsage]
    # No length in the message: the model's variants as they are.
    assert server._compared_variants(asked({}), wrong) == wrong  # pyright: ignore[reportPrivateUsage]
