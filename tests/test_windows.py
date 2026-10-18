"""The windows of the demo profiles against the ones PAC's own build_windows gives; their own
tests, on synthetic lines, are sigpipe's (sigpipe.masw.windows)."""

import json
import logging
from pathlib import Path
from typing import TypedDict

import pytest
from sigpipe.masw.profiles import load_profile
from sigpipe.masw.windows import (
    MASWParameters,
    MASWWindow,
    build_windows,
)

from paco.qc.coherence import near_field_windows
from paco.settings import Settings


# Format of tests/snapshots/pac_windows.json: the windows PAC's own build_windows gives for the
# test lines (masw.adapters.windows, PAC's until it took sigpipe's). "generated_with" in the file
# records the PAC and sigpipe commits used. The cases' distances suit active_p2's shots, 24 m apart:
# some windows keep a shot on each side, some none, and the narrowest band puts shots at exactly
# distance_min and distance_max, which leave them out.
class WindowReference(TypedDict):
    xmid: float
    files: list[str]
    receivers: list[int]
    sources: list[list[float]]


class CaseReference(TypedDict):
    profile: str
    params: dict[str, float]
    windows: list[WindowReference]


REFERENCE_CASES: list[CaseReference] = json.loads(
    (Path(__file__).parent / "snapshots" / "pac_windows.json").read_text()
)["cases"]


def _case_id(case: CaseReference) -> str:
    p = case["params"]
    return f"{case['profile']}-L{p['length']:g}-S{p['step']:g}-D{p['distance_min']:g}-{p['distance_max']:g}"


def _describe(window: MASWWindow) -> WindowReference:
    return {
        "xmid": window.xmid,
        "files": [path.name for path in window.selected_files],
        "receivers": [window.receiver_indices[0], window.receiver_indices[-1] + 1],
        "sources": [[a.source.x, a.source.z] for a in window.acquisitions],
    }


# ---------------------------------------------------------------- parity with PAC


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=_case_id)
def test_windows_match_pac(demo_settings: Settings, case: CaseReference) -> None:
    profile = load_profile(case["profile"], demo_settings)
    windows = build_windows(profile, MASWParameters.model_validate(case["params"]))

    assert len(windows) == len(case["windows"])
    for window, expected in zip(windows, case["windows"], strict=True):
        actual = _describe(window)
        assert actual["xmid"] == pytest.approx(expected["xmid"], abs=1e-9)
        assert {**actual, "xmid": 0.0} == {**expected, "xmid": 0.0}
        assert window.receiver_indices == list(range(*expected["receivers"]))


def test_reference_covers_skipped_windows() -> None:
    # Guards the reference itself: at least one case must drop windows for lack of valid shots.
    def n_windows(case: CaseReference) -> int:
        p = case["params"]
        return int((96 - p["length"]) // p["step"] + 1)

    assert any(len(case["windows"]) < n_windows(case) for case in REFERENCE_CASES)


# ---------------------------------------------------------------- behaviour


def test_passive_windows_use_the_first_receiver_as_source(demo_settings: Settings) -> None:
    profile = load_profile("passive_p2", demo_settings)
    params = MASWParameters(length=24, step=24, distance_min=0, distance_max=1)

    for window in build_windows(profile, params):
        assert len(window.selected_files) == 2
        assert all(a.source == a.receivers[0] for a in window.acquisitions)


def test_windows_without_valid_shots_are_skipped_with_a_warning(
    demo_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    profile = load_profile("active_p2", demo_settings)
    params = MASWParameters(length=24, step=6, distance_min=0.5, distance_max=20)

    with caplog.at_level(logging.WARNING, logger="paco.windows"):
        windows = build_windows(profile, params)

    assert len(windows) == 4
    assert sum("No valid shots" in message for message in caplog.messages) == 13 - 4


def test_length_longer_than_the_line_is_rejected(demo_settings: Settings) -> None:
    profile = load_profile("active_p2", demo_settings)
    params = MASWParameters(length=97, step=1, distance_min=0, distance_max=100)

    with pytest.raises(ValueError, match=r"length \(97\) exceeds the 96 receivers"):
        build_windows(profile, params)


def test_the_near_field_rule_counts_the_farther_shots_g1_keeps(demo_settings: Settings) -> None:
    # The first window (0 to 34.5 m) stacks 101.dat, 0.75 m before its first receiver, and three
    # shots beyond its end, from 12.75 m.
    profile = load_profile("active_p2", demo_settings)
    params = MASWParameters(length=24, step=24, distance_min=0, distance_max=100)
    first = build_windows(profile, params)[:1]
    far = ["133.dat", "149.dat", "165.dat"]

    def kept(excluded: tuple[str, ...]) -> tuple[list[str], int]:
        windows, near_only = near_field_windows(first, 10.0, excluded)
        return [path.name for path in windows[0].selected_files], near_only

    # A farther shot G1 keeps: the near one goes.
    assert kept(()) == (far, 0)
    assert kept(("133.dat", "149.dat")) == (far, 0)
    # Every farther record left out by G1: the window keeps its near shot, rather than none.
    assert kept(tuple(far)) == (["101.dat", *far], 1)
