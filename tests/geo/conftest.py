import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
from geo_judged import JudgedLine, make_judged_line
from local_lines import ACTIVE, PASSIVE, link_active, link_passive, present

from paco.agent import AgentSettings
from paco.geo.settings import GeoSettings
from paco.llm.settings import JudgeSettings, ModelSettings
from paco.modes import ModeSettings
from paco.settings import Settings, get_settings

# active_p2's shots the tests keep: the seven at each end, which a long window stacks, and one in
# six between, 28 of 97: real records, read and judged in a quarter of the time.
END_SHOTS = 7
SHOT_STEP = 6


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The settings from the environment alone, never from a .env in the folder the tests run
    from: a developer's own (their folders, their ports, their model) would make the tests pass
    where a fresh clone fails."""
    for settings in (
        Settings,
        AgentSettings,
        GeoSettings,
        ModelSettings,
        JudgeSettings,
        ModeSettings,
    ):
        monkeypatch.setitem(settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def input_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """active_p2, its records linked from the local data folder with the shots the tests keep, in
    a folder of its own; passive_p2 beside it, whole (its 16 records of noise process in seconds)."""
    present(ACTIVE, PASSIVE)
    root = tmp_path_factory.mktemp("input")
    link_active(root, END_SHOTS, SHOT_STEP)
    link_passive(root)
    return root


@pytest.fixture(scope="session")
def passive_dir(input_dir: Path) -> Path:
    """The test lines' folder, holding passive_p2."""
    return input_dir


@pytest.fixture
def paco_env(
    input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    """PACo's settings, as the server's environment gives them: the test lines, and the outputs in
    `tmp_path`."""
    monkeypatch.setenv("PACO_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("PACO_OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def free_port() -> int:
    """A port of this machine that nothing listened on a moment ago."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="session")
def judged_line(input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> JudgedLine:
    """A line of active_p2, processed, picked and judged once for the tests that read real curves
    and images."""
    return make_judged_line(input_dir, tmp_path_factory.mktemp("judged_line"))
