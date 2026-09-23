import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from paco.settings import Settings

# PACo reads PAC's demo profiles in place, so the PAC clone must sit next to the PACo clone.
PAC_INPUT_DIR = Path(__file__).resolve().parents[2] / "PAC" / "data" / "input"

type CopyDemo = Callable[[str, str], Path]


@pytest.fixture(scope="session")
def demo_input_dir() -> Path:
    if not (PAC_INPUT_DIR / "active_p1").is_dir():
        pytest.skip(f"PAC demo profiles not found in {PAC_INPUT_DIR}")
    return PAC_INPUT_DIR


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
