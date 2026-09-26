import json
import logging
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import ValidationError
from sigpipe.base import Coordinate, LinearAcquisition

from paco.profiles import load_profile
from paco.settings import Settings
from paco.windows import Exclusions, MASWParameters, MASWWindow, apply_exclusions, build_windows


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"length": 2},
        {"step": 0},
        {"distance_min": -1.0},
        {"distance_max": 0.0},
        {"distance_min": 5.0, "distance_max": 5.0},
    ],
)
def test_invalid_masw_parameters(overrides: dict[str, float]) -> None:
    valid = {"length": 24, "step": 12, "distance_min": 1.0, "distance_max": 10.0}

    with pytest.raises(ValidationError):
        MASWParameters.model_validate(valid | overrides)


def test_each_record_leaves_out_its_own_traces_on_active_windows() -> None:
    receivers = tuple(Coordinate(float(x), 0.0, 0.0) for x in range(6))

    def acquisition(source: float) -> LinearAcquisition:
        return LinearAcquisition(source=Coordinate(source, 0.0, 0.0), receivers=receivers)

    window = MASWWindow(
        xmid=2.5,
        selected_files=[Path("a.dat"), Path("b.dat"), Path("c.dat")],
        receiver_indices=list(range(6)),
        acquisitions=[acquisition(-1.0), acquisition(-2.0), acquisition(7.0)],
    )
    # Trace 1 is a.dat's own; trace 4 two of the three records excluded: a receiver's defect.
    exclusions = Exclusions(traces={"a.dat": (1, 4), "b.dat": (4,)})

    # One geometry (passive): a trace any record excluded leaves every record.
    union = apply_exclusions(window, exclusions)
    assert union is not None and union.receiver_indices == [0, 2, 3, 5]
    assert union.record_receivers is None
    # Each its own (active): the window keeps its receivers, each record gives its own.
    own = apply_exclusions(window, exclusions, "per_record")
    assert own is not None and own.receiver_indices == list(range(6))
    assert own.record_receivers == [[0, 2, 3, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]
    assert [len(one.receivers) for one in own.acquisitions] == [4, 5, 5]
    # One set of receivers (passive-active, correlation gathers stacked): trace 4 leaves every
    # record, and a.dat, which excluded trace 1 too, leaves the window.
    shared = apply_exclusions(window, exclusions, "shared")
    assert shared is not None and shared.receiver_indices == [0, 1, 2, 3, 5]
    assert [path.name for path in shared.selected_files] == ["b.dat", "c.dat"]
    assert shared.record_receivers is None
    assert [len(one.receivers) for one in shared.acquisitions] == [5, 5]
    # Two records: every exclusion is half of them, so it is the union again.
    two = window.model_copy(
        update={
            "selected_files": window.selected_files[:2],
            "acquisitions": window.acquisitions[:2],
        }
    )
    both = apply_exclusions(two, Exclusions(traces={"a.dat": (1,)}), "per_record")
    assert both is not None and both.record_receivers == [[0, 2, 3, 4, 5]] * 2
