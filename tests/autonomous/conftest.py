import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
from autonomous_judged import JudgedLine, make_judged_line
from local_lines import ACTIVE, PASSIVE, PROFILE_SHOTS, link_active, link_passive, present

from paco.agent import AgentSettings
from paco.agent.settings import OutputSettings
from paco.autonomous.tools import search as searching
from paco.autonomous.tools.trials import Study
from paco.llm.settings import ModelSettings
from paco.modes import ModeSettings
from paco.settings import Settings, get_settings

# The active test profile's 25 shots; the passive one's first 2 records of noise.
SHOTS = PROFILE_SHOTS
NOISE_RECORDS = 2


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The settings from the environment alone, never from a .env in the folder the tests run
    from: a developer's own (their folders, their ports, their model) would make the tests pass
    where a fresh clone fails."""
    for settings in (
        Settings,
        AgentSettings,
        ModelSettings,
        OutputSettings,
        ModeSettings,
    ):
        monkeypatch.setitem(settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def input_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The test profiles linked from PAC's data folder, with the shots and records the tests keep,
    in a folder of their own."""
    present(ACTIVE, PASSIVE)
    root = tmp_path_factory.mktemp("input")
    link_active(root, SHOTS)
    link_passive(root, NOISE_RECORDS)
    return root


@pytest.fixture(scope="session")
def passive_dir(input_dir: Path) -> Path:
    """The test lines' folder, holding passive_test_profile."""
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


@pytest.fixture(scope="module")
def searched() -> Iterator[None]:
    """Every study taken as searched and its sheets judged, for the scripted plays that try and
    commit settings without the full grid (minutes long); the module's later tests too."""

    def judged(_study: Study) -> None:
        return None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(searching, "unjudged", judged)
        yield


@pytest.fixture
def free_port() -> int:
    """A port of this machine that nothing listened on a moment ago."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="session")
def judged_line(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> JudgedLine:
    """A line of active_test_profile, processed, picked and judged once for the tests that read real curves
    and images: on the 7 shots PACo's own tests keep (tests/conftest.py), the line their test_qc.py
    judges."""
    return make_judged_line(demo_input_dir, tmp_path_factory.mktemp("judged_line"))
