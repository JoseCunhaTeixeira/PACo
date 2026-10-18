"""The real lines the tests read, kept locally and in no repository: PAC's own data folder
(../PAC/data/input), or the one PACO_TEST_INPUT_DIR names. active_p2: 96 receivers 1.5 m apart and
97 shots of 3 s; passive_p2: 16 records of ambient noise, 120 s long, on the same receivers. Each
suite links the part of them it reads into a folder of its own."""

import os
from pathlib import Path

import pytest
import yaml

LOCAL_INPUT_DIR = Path(
    os.environ.get("PACO_TEST_INPUT_DIR")
    or Path(__file__).resolve().parents[2] / "PAC" / "data" / "input"
)
ACTIVE = "active_p2"
PASSIVE = "passive_p2"


def present(*names: str) -> None:
    """Skip, saying why, when a line is not in the local data folder."""
    for name in names:
        if not (LOCAL_INPUT_DIR / name / "receiver_positions.yaml").is_file():
            pytest.skip(f"{name} is not in {LOCAL_INPUT_DIR}: set PACO_TEST_INPUT_DIR")


def link_active(root: Path, ends: int, step: int) -> Path:
    """active_p2 linked into `root` with the shots kept: `ends` at each end of the line, and one
    in `step` between them."""
    source, target = LOCAL_INPUT_DIR / ACTIVE, root / ACTIVE
    target.mkdir()
    shots: dict[str, object] = yaml.safe_load((source / "source_positions.yaml").read_text())
    names = sorted(shots)
    kept = sorted(
        {*names[:ends], *names[len(names) - ends :], *names[ends : len(names) - ends : step]}
    )
    (target / "receiver_positions.yaml").symlink_to(source / "receiver_positions.yaml")
    (target / "source_positions.yaml").write_text(
        yaml.safe_dump({name: shots[name] for name in kept})
    )
    for name in kept:
        (target / name).symlink_to(source / name)
    return target


def link_passive(root: Path, records: int | None = None) -> Path:
    """passive_p2 linked into `root`: its first `records` records, all of them when None."""
    source, target = LOCAL_INPUT_DIR / PASSIVE, root / PASSIVE
    target.mkdir()
    noise = sorted(path for path in source.iterdir() if path.suffix != ".yaml")
    for path in [*source.glob("*.yaml"), *noise[:records]]:
        (target / path.name).symlink_to(path)
    return target
