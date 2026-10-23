"""S2's segments on a passive line: their length, set by the window's span, and their FK
selection, judged by the M0 pick their images give, the correlograms converged."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.base import Coordinate, LinearAcquisition, Stream
from sigpipe.masw.presets import make_preset, resolve_preset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.runs.processing import preprocess_records
from sigpipe.masw.windows import Exclusions

from paco.qc import segments
from paco.qc.g1_signal import SignalThresholds
from paco.qc.segments import SEGMENTS_FILE, SegmentRules, choose_segments

WINDOWS = {"masw": {"length": 24, "step": 24}}
SAMPLING = 500.0


@pytest.mark.slow
def test_the_demos_passive_line_tries_the_fk_selection_of_its_segments(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    # passive_test_profile's windows span 34.5 m, crossed in 0.43 s at 80 m/s: segments of ten
    # crossings, 4.31 s, as the ladder sizes them. Their FK selection at 0.05 gives an M0 pick over
    # 4.3 times its shortest wavelength, 77 % of the segments kept and the correlograms converged
    # (0.72), against 3.9 with the line's own (0.2), whose correlograms do not converge (0.41).
    profile = profiles["passive_test_profile"]
    rules = SegmentRules()
    sized = segments.sized_slicing(24, profile, rules, 80.0)
    preset = resolve_preset(make_preset("passive", {**WINDOWS, **sized}), profile)
    preprocess_records(preset, profile, tmp_path, workers=1)

    changes, notes = choose_segments(
        profile, preset, tmp_path, rules, SignalThresholds(), PickingParameters(), Exclusions()
    )

    assert changes == {"selection": {"method": "fk", "threshold": 0.05, "vmin": None, "vmax": None}}
    (note,) = notes
    assert note.startswith(
        "FK selection at 0.05 at any velocity on the line's 4.31 s segments: M0 picked over 4.3 "
        "times its shortest wavelength against 3.9 with the line's own selection"
    )
    trials = json.loads((tmp_path / SEGMENTS_FILE).read_text())
    # The FK selection always on: every band and threshold tried, the line's own among them, on
    # the line's segments.
    assert len(trials["candidates"]) == len(rules.bands) * len(rules.thresholds)
    assert {one["segment_s"] for one in trials["candidates"]} == {4.31}
    assert trials["chosen"]["convergence"] >= rules.min_convergence
    # The user's own selection stays as it is.
    kept, _ = choose_segments(
        profile, preset, tmp_path, rules, SignalThresholds(), PickingParameters(),
        Exclusions(), frozenset({"slicing", "selection"}),
    )  # fmt: skip
    assert kept is None


def _shot(seed: int | None) -> Stream:
    """A virtual shot, 12 traces 1 m apart: a wavelet arriving at 200 m/s, with noise drawn from
    `seed` (None: none)."""
    ts = np.arange(int(SAMPLING)) / SAMPLING
    receivers = tuple(Coordinate(float(i), 0.0, 0.0) for i in range(12))
    acquisition = LinearAcquisition(source=receivers[0], receivers=receivers)
    xt = np.zeros((12, ts.size))
    for i, offset in enumerate(acquisition.offsets):
        tau = ts - offset / 200.0
        xt[i] = np.where(tau >= 0, np.exp(-40 * tau) * np.cos(2 * np.pi * 20 * tau), 0)
    if seed is not None:
        xt += np.random.default_rng(seed).standard_normal(xt.shape) * 2
    return Stream(xt=xt.astype(np.float32), ts=ts.astype(np.float32),
                  sampling_freq=SAMPLING, acquisition=acquisition)  # fmt: skip


def test_converged_halves_agree_over_their_arrivals() -> None:
    # Two halves both holding the wave agree; halves of noise alone do not.
    signal = SignalThresholds()

    agreed = segments.halves_agreement(_shot(None), _shot(None), signal)
    noisy = segments.halves_agreement(_shot(1), _shot(2), signal)

    assert agreed == pytest.approx(1.0)
    assert noisy is not None and abs(noisy) < 0.3


def test_the_length_trial_slices_each_length_as_its_window_asks() -> None:
    # 15 receivers 1.5 m apart, 21 m: at 80 m/s ten crossings make 2.62 s; 61 receivers would ask
    # 11.25 s, held to the shortest record's 10 s.
    profile = SimpleNamespace(
        receivers=[SimpleNamespace(x=1.5 * k) for k in range(96)],
        records=[SimpleNamespace(duration_s=10.0)],
    )

    def seconds(length: int) -> object:
        sliced = segments.sized_slicing(length, profile, SegmentRules(), 80.0)  # pyright: ignore[reportArgumentType]
        return sliced["slicing"]

    assert seconds(15) == {"segment_duration": 2.62, "segment_step": 2.62}
    assert seconds(61) == {"segment_duration": 10.0, "segment_step": 10.0}
