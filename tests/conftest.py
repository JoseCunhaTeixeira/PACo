import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib
import pytest
from local_lines import ACTIVE, PASSIVE, link_active, link_passive, present
from sigpipe.masw.profiles import Profile, load_profile

from paco.agent.settings import AgentSettings, OutputSettings
from paco.llm.settings import ModelSettings
from paco.modes import ModeSettings
from paco.settings import Settings, get_settings

# Pipelines save figures to files: use the non-GUI backend, as PAC's API does.
matplotlib.use("Agg")

# The real lines PACo's tests read (local_lines): active_p2 with 7 of its shots, one in 16 from
# the line's first to its last, 24 m apart, each window stacking a few, as the demo line's 2 shots
# did; passive_p2 with its first 2 records of noise. The autonomous agent's tests read more of them
# (tests/autonomous/conftest.py).
DEMO_PROFILES = (ACTIVE, PASSIVE)
END_SHOTS = 0
SHOT_STEP = 16
NOISE_RECORDS = 2

type CopyDemo = Callable[[str, str], Path]


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The settings from the environment alone, never from a `.env` in the folder the tests run
    from: a developer's own (their input folder, their model's context) would make the tests
    pass where CI, which has none, fails."""
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
def demo_input_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The test lines, linked from the local data folder into a folder of their own: profiles the
    user adds to that folder stay out of the tests."""
    present(*DEMO_PROFILES)
    root = tmp_path_factory.mktemp("input")
    link_active(root, END_SHOTS, SHOT_STEP)
    link_passive(root, NOISE_RECORDS)
    return root


@pytest.fixture(scope="session")
def profiles(demo_input_dir: Path) -> dict[str, Profile]:
    """The demo profiles, loaded once: profiles are frozen, so tests can share them."""
    settings = Settings(input_dir=demo_input_dir)
    return {name: load_profile(name, settings) for name in DEMO_PROFILES}


@pytest.fixture
def demo_settings(demo_input_dir: Path, tmp_path: Path) -> Settings:
    return Settings(input_dir=demo_input_dir, output_dir=tmp_path / "output")


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    path = tmp_path / "input"
    path.mkdir()
    return path


@pytest.fixture
def settings(input_dir: Path, tmp_path: Path) -> Settings:
    return Settings(input_dir=input_dir, output_dir=tmp_path / "output")


@pytest.fixture
def copy_demo(demo_input_dir: Path, input_dir: Path) -> CopyDemo:
    """Copy a demo profile into `input_dir` under a new name, to be broken by a test.

    YAML files are copied so tests can edit them; records are symlinked to avoid copying
    tens of MB per test.
    """

    def copy(demo: str, name: str) -> Path:
        destination = input_dir / name
        destination.mkdir()
        for path in (demo_input_dir / demo).iterdir():
            if path.suffix == ".yaml":
                shutil.copy(path, destination / path.name)
            else:
                (destination / path.name).symlink_to(path)
        return destination

    return copy


@pytest.fixture
def paco_env(
    demo_input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    """The server's settings, as its environment gives them: the demo profiles, outputs in
    `tmp_path`, and one inversion retry a window (the tests' short
    samplers never hold the effective samples G5 asks, and each retry doubles them)."""
    monkeypatch.setenv("PACO_INPUT_DIR", str(demo_input_dir))
    monkeypatch.setenv("PACO_OUTPUT_DIR", str(tmp_path / "output"))
    qc_config = tmp_path / "qc_config.json"
    qc_config.write_text('{"budgets": {"inversion_per_window": 1}}')
    monkeypatch.setenv("PACO_QC_CONFIG", str(qc_config))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()
