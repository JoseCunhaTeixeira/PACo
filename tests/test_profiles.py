from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from paco.profiles import (
    RECEIVER_POSITIONS_FILE,
    SOURCE_POSITIONS_FILE,
    ProfileError,
    ProfileKind,
    inspect_profile,
    list_profiles,
    load_profile,
)
from paco.settings import Settings
from sigpipe.base import Coordinate

# Signature of the `copy_demo` fixture from conftest.py.
type CopyDemo = Callable[[str, str], Path]

# ---------------------------------------------------------------- demo profiles


def test_list_profiles_finds_the_demo_profiles(demo_settings: Settings) -> None:
    assert {"active_p1", "passive_p1"} <= set(list_profiles(demo_settings))


def test_list_profiles_skips_files_and_hidden_folders(input_dir: Path, settings: Settings) -> None:
    (input_dir / "p1").mkdir()
    (input_dir / ".hidden").mkdir()
    (input_dir / "notes.txt").write_text("not a profile")

    assert list_profiles(settings) == ["p1"]


def test_inspect_active_demo(demo_settings: Settings) -> None:
    summary = inspect_profile("active_p1", demo_settings)

    assert summary.kind == ProfileKind.ACTIVE
    assert summary.n_records == 2
    assert summary.n_receivers == 96
    assert summary.receiver_x_range_m == (0.0, 23.75)
    assert summary.receiver_spacing_m == pytest.approx(0.25)
    assert summary.sampling_rate_hz == 2000.0
    assert summary.nyquist_hz == 1000.0
    assert summary.record_duration_range_s == pytest.approx((2.0, 2.0), abs=1e-3)
    assert summary.source_x_range_m == (-0.75, 24.5)


def test_inspect_passive_demo(demo_settings: Settings) -> None:
    summary = inspect_profile("passive_p1", demo_settings)

    assert summary.kind == ProfileKind.PASSIVE
    assert summary.n_records == 2
    assert summary.n_receivers == 96
    assert summary.sampling_rate_hz == 500.0
    assert summary.nyquist_hz == 250.0
    assert summary.record_duration_range_s == pytest.approx((89.998, 129.998), abs=1e-3)
    assert summary.source_x_range_m is None


@pytest.mark.parametrize("name", ["active_p1", "passive_p1"])
def test_summary_stays_short(demo_settings: Settings, name: str) -> None:
    # The summary is what the agent reads: it must stay far below the tool-output budget.
    assert len(inspect_profile(name, demo_settings).model_dump_json()) < 1_000


def test_load_active_demo_uses_sigpipe_coordinates(demo_settings: Settings) -> None:
    profile = load_profile("active_p1", demo_settings)

    assert all(isinstance(receiver, Coordinate) for receiver in profile.receivers)
    assert all(receiver.y == 0.0 for receiver in profile.receivers)
    assert [record.path.name for record in profile.records] == ["1.dat", "2.dat"]
    assert profile.records[0].source == Coordinate(x=-0.75, y=0.0, z=0.0)
    assert profile.records[1].source == Coordinate(x=24.5, y=0.0, z=0.0)
    assert profile.records[0].n_traces == 96
    assert profile.records[0].duration_s == pytest.approx(1.9995, abs=1e-6)


# ---------------------------------------------------------------- valid variants


def test_profile_without_source_file_is_passive(copy_demo: CopyDemo, settings: Settings) -> None:
    folder = copy_demo("active_p1", "p1")
    (folder / SOURCE_POSITIONS_FILE).unlink()

    profile = load_profile("p1", settings)

    assert profile.kind == ProfileKind.PASSIVE
    assert all(record.source is None for record in profile.records)


# ---------------------------------------------------------------- errors


def test_unknown_profile_lists_the_available_ones(copy_demo: CopyDemo, settings: Settings) -> None:
    copy_demo("active_p1", "p1")

    with pytest.raises(ProfileError, match=r"Unknown profile 'nope'\. Available profiles: p1\."):
        load_profile("nope", settings)


@pytest.mark.parametrize("name", ["../input", "p1/..", "/tmp"])
def test_names_outside_the_input_dir_are_rejected(
    copy_demo: CopyDemo, settings: Settings, name: str
) -> None:
    copy_demo("active_p1", "p1")

    with pytest.raises(ProfileError, match="Unknown profile"):
        load_profile(name, settings)


def test_profile_without_records(copy_demo: CopyDemo, settings: Settings) -> None:
    folder = copy_demo("active_p1", "p1")
    for record in ("1.dat", "2.dat"):
        (folder / record).unlink()

    with pytest.raises(ProfileError, match="has no record files"):
        load_profile("p1", settings)


def test_missing_receiver_file(copy_demo: CopyDemo, settings: Settings) -> None:
    folder = copy_demo("active_p1", "p1")
    (folder / RECEIVER_POSITIONS_FILE).unlink()

    with pytest.raises(ProfileError, match=f"has no {RECEIVER_POSITIONS_FILE}"):
        load_profile("p1", settings)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("x: [unclosed", "is not valid YAML"),
        ("x: 0.0\nz: 0.0\n", "must be a list"),
        ("- x: 0.0\n  z: 0.0\n- x: 1.0\n", "needs numeric 'x' and 'z'"),
        ("- x: 0.0\n  z: 0.0\n- x: one\n  z: 0.0\n", "needs numeric 'x' and 'z'"),
        ("- x: 0.0\n  z: 0.0\n", "needs at least 2 receivers, found 1"),
    ],
)
def test_malformed_receiver_file(
    copy_demo: CopyDemo, settings: Settings, content: str, message: str
) -> None:
    folder = copy_demo("active_p1", "p1")
    (folder / RECEIVER_POSITIONS_FILE).write_text(content)

    with pytest.raises(ProfileError, match=message):
        load_profile("p1", settings)


def test_unsorted_receivers(copy_demo: CopyDemo, settings: Settings) -> None:
    folder = copy_demo("active_p1", "p1")
    path = folder / RECEIVER_POSITIONS_FILE
    path.write_text(yaml.safe_dump(yaml.safe_load(path.read_text())[::-1]))

    with pytest.raises(ProfileError, match="must be sorted by x"):
        load_profile("p1", settings)


def test_source_file_not_matching_the_records(copy_demo: CopyDemo, settings: Settings) -> None:
    folder = copy_demo("active_p1", "p1")
    (folder / SOURCE_POSITIONS_FILE).write_text(
        '"1.dat": {x: -0.75, z: 0.0}\n"3.dat": {x: 30.0, z: 0.0}\n'
    )

    with pytest.raises(
        ProfileError,
        match=r"no source position for 2\.dat; positions for unknown records 3\.dat",
    ):
        load_profile("p1", settings)


def test_source_file_that_is_not_a_mapping(copy_demo: CopyDemo, settings: Settings) -> None:
    folder = copy_demo("active_p1", "p1")
    (folder / SOURCE_POSITIONS_FILE).write_text("- x: 0.0\n  z: 0.0\n")

    with pytest.raises(ProfileError, match="must map each record file name"):
        load_profile("p1", settings)


def test_fewer_receivers_than_traces(copy_demo: CopyDemo, settings: Settings) -> None:
    folder = copy_demo("active_p1", "p1")
    path = folder / RECEIVER_POSITIONS_FILE
    path.write_text(yaml.safe_dump(yaml.safe_load(path.read_text())[:10]))

    with pytest.raises(ProfileError, match=r"cannot load record 1\.dat.*lists 10 receivers"):
        load_profile("p1", settings)


def test_file_sigpipe_cannot_read(copy_demo: CopyDemo, settings: Settings) -> None:
    folder = copy_demo("active_p1", "p1")
    (folder / SOURCE_POSITIONS_FILE).unlink()
    (folder / "notes.txt").write_text("field notes")

    with pytest.raises(ProfileError, match=r"sigpipe cannot load record notes\.txt"):
        load_profile("p1", settings)


def test_records_with_different_sampling_rates(
    copy_demo: CopyDemo, demo_input_dir: Path, settings: Settings
) -> None:
    folder = copy_demo("active_p1", "p1")
    (folder / SOURCE_POSITIONS_FILE).unlink()
    # A 500 Hz passive record next to the 2000 Hz active ones, same 96 receivers.
    (folder / "3.dat").symlink_to(demo_input_dir / "passive_p1" / "2.dat")

    with pytest.raises(ProfileError, match="one sampling rate, found 500 Hz, 2000 Hz"):
        load_profile("p1", settings)
