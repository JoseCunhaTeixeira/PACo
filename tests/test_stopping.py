"""Stopping the agent's work on request (paco.stopping): a call keeps the stop of the answer it
runs in and a job takes it along, a new run stopped is removed whole, a redo stopped gives each
window back what it had, and an answer stopped midway leaves a conversation the model can read."""

import contextvars
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from mcp import Client
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from sigpipe.masw.runs import Stopped, find_run, run_processing

from paco import stopping
from paco.agent.loop import STOPPED, Agent
from paco.agent.model import Filled, Reply, ToolCall
from paco.qc import QCConfig, invalidate, process_line, read_attempts, rerun_phase_shift
from paco.qc.attempts import restore
from paco.settings import Settings

# The scope of a message that only looks at what exists.
LOOK = {
    "process": False,
    "pick": False,
    "invert": False,
    "soils": False,
    "profile": None,
    "run_id": None,
    "positions_m": [],
    "redo": False,
    "replace_hand_work": False,
    "option": None,
}

SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}


@pytest.fixture
def stopped() -> Iterator[threading.Event]:
    """This test's work stopped from its start, as a stopped answer's is."""
    signal = stopping.Signal()
    signal.stop()
    token = stopping.SIGNAL.set(signal)
    yield signal.event
    stopping.SIGNAL.reset(token)


@pytest.fixture(scope="module")
def processed(
    demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Settings, str]:
    """active_p1 processed once, in four 24-receiver windows."""
    root = tmp_path_factory.mktemp("stopping")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        manifest = run_processing("active_p1", "active", SMALL_WINDOWS, settings)
    return settings, manifest.run_id


def test_a_call_keeps_the_stop_of_its_answer_and_a_job_takes_it_along() -> None:
    signal = stopping.Signal()
    token = stopping.SIGNAL.set(signal)
    try:
        first = signal.event

        def call() -> threading.Event | None:
            stopping.pin()
            signal.renew()  # the next answer, while this call still runs
            return stopping.current()

        # A call pins its answer's stop: the next answer's cannot restart it.
        assert contextvars.copy_context().run(call) is first
        signal.stop()
        job = stopping.bound(stopping.current)
        seen: list[threading.Event | None] = []
        thread = threading.Thread(target=lambda: seen.append(job()))
        thread.start()
        thread.join()
        assert seen == [signal.event] and signal.event.is_set()
    finally:
        stopping.SIGNAL.reset(token)
    assert stopping.current() is None  # outside any answer, nothing stops
    stopping.check()


def test_a_window_redone_is_given_back_what_it_had(tmp_path: Path) -> None:
    window = tmp_path / "xmid_12.50"
    window.mkdir()
    before = {
        "window.json": "the window",
        "DispersionImage_0000.hdf5": "S2",
        "DispersionCurves_0000.csv": "S3",
        "quality.json": "S3",
    }
    for name, content in before.items():
        (window / name).write_text(content)

    invalidate(window, "phase_shift", attempt=1)
    (window / "DispersionImage_0000.hdf5").write_text("half of a new image")  # a stopped redo's
    restore(window, "phase_shift", attempt=1)

    assert {path.name: path.read_text() for path in window.iterdir()} == before


def test_a_run_stopped_is_removed_whole(
    demo_input_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stopped: threading.Event,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(input_dir=demo_input_dir, output_dir=tmp_path / "outputs", workers=2)

    with pytest.raises(Stopped):
        process_line("active_p1", SMALL_WINDOWS, settings, QCConfig())

    assert stopped.is_set()
    assert not list((tmp_path / "outputs").glob("active_p1/*"))


def test_a_redo_stopped_gives_each_window_back_what_it_had(
    processed: tuple[Settings, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stopped: threading.Event,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings, run_id = processed
    run_folder = find_run(run_id, settings)
    windows = ["xmid_2.88", "xmid_8.88"]
    files = {unit: sorted(path.name for path in (run_folder / unit).iterdir()) for unit in windows}
    images = {
        unit: (run_folder / unit / "DispersionImage_0000.hdf5").read_bytes() for unit in windows
    }

    with pytest.raises(Stopped):
        rerun_phase_shift(run_id, windows, {"dispersion": {"vmax": 500}}, settings)

    assert stopped.is_set()
    for unit in windows:
        assert sorted(path.name for path in (run_folder / unit).iterdir()) == files[unit]
        assert (run_folder / unit / "DispersionImage_0000.hdf5").read_bytes() == images[unit]
    # Nothing redone, nothing logged but the run's first attempts.
    assert all(
        attempt.attempt == 1
        for attempt in read_attempts(run_folder)
        if attempt.stage == "phase_shift"
    )


class _Endless:
    """A tool that never answers, as a long one does while the user stops it."""

    async def call_tool(self, _name: str, _arguments: object, **_options: object) -> None:
        await anyio.sleep_forever()


def test_an_answer_stopped_midway_leaves_a_conversation_the_model_can_read() -> None:
    replies = [
        Reply(
            content="",
            tool_calls=(ToolCall(id="call_0", name="inspect", arguments='{"what": "runs"}'),),
        )
    ]

    class Model:
        async def __call__(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[ChatCompletionFunctionToolParam],
        ) -> Reply:
            del messages, tools  # the reply is scripted
            return replies.pop(0)

        async def fill(
            self, messages: list[ChatCompletionMessageParam], schema: dict[str, Any]
        ) -> Filled:
            del messages, schema  # a message that only looks
            return Filled(content=json.dumps({**LOOK}))

    agent = Agent(cast(Client, _Endless()), Model(), [], None, on_event=lambda _: None)

    async def ask() -> None:
        with anyio.move_on_after(0.5):
            await agent.answer("Which profiles are there?")

    anyio.run(ask)

    # The call it had asked for, and its unanswered request, gone: the question and one line.
    assert agent.messages[1:] == [
        {"role": "user", "content": "Which profiles are there?"},
        {"role": "assistant", "content": STOPPED},
    ]
