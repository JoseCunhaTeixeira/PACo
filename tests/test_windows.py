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

from paco.settings import Settings


# Format of tests/data/pac_windows.json: the windows PAC's own build_windows gives for the demo
# profiles. "generated_with" in the file records the PAC and sigpipe commits used.
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
    (Path(__file__).parent / "data" / "pac_windows.json").read_text()
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
    profile = load_profile("passive_p1", demo_settings)
    params = MASWParameters(length=24, step=24, distance_min=0, distance_max=1)

    for window in build_windows(profile, params):
        assert len(window.selected_files) == 2
        assert all(a.source == a.receivers[0] for a in window.acquisitions)


def test_windows_without_valid_shots_are_skipped_with_a_warning(
    demo_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    profile = load_profile("active_p1", demo_settings)
    params = MASWParameters(length=24, step=6, distance_min=0.5, distance_max=5)

    with caplog.at_level(logging.WARNING, logger="paco.windows"):
        windows = build_windows(profile, params)

    assert len(windows) == 2
    assert sum("No valid shots" in message for message in caplog.messages) == 13 - 2


def test_length_longer_than_the_line_is_rejected(demo_settings: Settings) -> None:
    profile = load_profile("active_p1", demo_settings)
    params = MASWParameters(length=97, step=1, distance_min=0, distance_max=100)

    with pytest.raises(ValueError, match=r"length \(97\) exceeds the 96 receivers"):
        build_windows(profile, params)
