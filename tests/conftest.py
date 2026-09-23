import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib
import pytest

from paco.profiles import Profile, load_profile
from paco.settings import Settings, get_settings

# Pipelines save figures to files: use the non-GUI backend, as PAC's API does.
matplotlib.use("Agg")

# PAC's demo profiles, as PACo's repository keeps a copy of them: the tests need no PAC clone.
DEMO_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "input"

type CopyDemo = Callable[[str, str], Path]


@pytest.fixture(scope="session")
def demo_input_dir() -> Path:
    if not (DEMO_INPUT_DIR / "active_p1").is_dir():
        pytest.skip(f"Demo profiles not found in {DEMO_INPUT_DIR}")
    return DEMO_INPUT_DIR


@pytest.fixture(scope="session")
def profiles(demo_input_dir: Path) -> dict[str, Profile]:
    """The demo profiles, loaded once: profiles are frozen, so tests can share them."""
    settings = Settings(input_dir=demo_input_dir)
    return {name: load_profile(name, settings) for name in ("active_p1", "passive_p1")}


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
    """The server's settings, as its environment gives them: the demo profiles, and outputs in
    `tmp_path` rather than data/output."""
    monkeypatch.setenv("PACO_INPUT_DIR", str(demo_input_dir))
    monkeypatch.setenv("PACO_OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()
