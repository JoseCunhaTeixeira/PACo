"""The lines the tests read: PAC's test profiles, in PAC's data folder (../PAC/data/input), or the
folder PACO_TEST_INPUT_DIR names. active_test_profile: 96 receivers 1.5 m apart and 25 shots of
3 s, 6 m apart; passive_test_profile: 10 records of ambient noise, 120 s long, on the same
receivers. Each suite links the shots and records it reads into a folder of its own, by name: the
same data from a folder holding the whole lines."""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

LOCAL_INPUT_DIR = Path(
    os.environ.get("PACO_TEST_INPUT_DIR")
    or Path(__file__).resolve().parents[2] / "PAC" / "data" / "input"
)
ACTIVE = "active_test_profile"
PASSIVE = "passive_test_profile"
# The active test profile's 25 shots, 6 m apart: one in four of the real line's 97.
PROFILE_SHOTS = tuple(f"{shot}.dat" for shot in range(101, 198, 4))


def present(*names: str) -> None:
    """Skip, saying why, when a line is not in the data folder."""
    for name in names:
        if not (LOCAL_INPUT_DIR / name / "receiver_positions.yaml").is_file():
            pytest.skip(f"{name} is not in {LOCAL_INPUT_DIR}: set PACO_TEST_INPUT_DIR")


def link_active(root: Path, shots: Sequence[str]) -> Path:
    """The active test profile linked into `root`, with the records `shots` names."""
    source, target = LOCAL_INPUT_DIR / ACTIVE, root / ACTIVE
    target.mkdir()
    positions: dict[str, object] = yaml.safe_load((source / "source_positions.yaml").read_text())
    (target / "receiver_positions.yaml").symlink_to(source / "receiver_positions.yaml")
    (target / "source_positions.yaml").write_text(
        yaml.safe_dump({name: positions[name] for name in shots})
    )
    for name in shots:
        (target / name).symlink_to(source / name)
    return target


def link_passive(root: Path, records: int) -> Path:
    """The passive test profile linked into `root`: its first `records` records."""
    source, target = LOCAL_INPUT_DIR / PASSIVE, root / PASSIVE
    target.mkdir()
    noise = sorted(path for path in source.iterdir() if path.suffix != ".yaml")
    for path in [*source.glob("*.yaml"), *noise[:records]]:
        (target / path.name).symlink_to(path)
    return target
