"""S2's segments on a passive line: their length, set by the window's span, and the FK selection,
judged by the M0 pick their images give, the correlograms converged."""

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


def test_the_demos_passive_line_takes_shorter_segments(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    # passive_p1's windows span 5.75 m, crossed in 72 ms at 80 m/s: 0.72 s segments (ten
    # crossings) with the FK selection at 0.1 give an M0 pick over 7.8 times its shortest
    # wavelength, 8 % of the segments kept and the correlograms converged (0.85), against 5.0
    # with the defaults (1 s, FK at 0.2), whose correlograms do not converge (0.48).
    profile = profiles["passive_p1"]
    preset = resolve_preset(make_preset("passive", WINDOWS), profile)
    preprocess_records(preset, profile, tmp_path, workers=1)
    rules = SegmentRules()

    changes, notes = choose_segments(
        profile, preset, tmp_path, rules, SignalThresholds(), PickingParameters(), Exclusions()
    )

    assert changes == {
        "slicing": {"segment_duration": 0.72, "segment_step": 0.72},
        "selection": {"method": "fk", "threshold": 0.1, "vmin": None, "vmax": None},
    }
    (note,) = notes
    assert note.startswith("Segments of 0.72 s (10 crossings of the windows' 5.75 m at 80 m/s)")
    trials = json.loads((tmp_path / SEGMENTS_FILE).read_text())
    # The FK selection always on: every band and threshold tried, the line's own among them.
    selections = len(rules.bands) * len(rules.thresholds)
    lengths = {one["segment_s"] for one in trials["candidates"]}
    assert len(trials["candidates"]) == len(lengths) * selections
    assert trials["chosen"]["convergence"] >= rules.min_convergence
    # The user's own settings stay as they are.
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


def test_the_lengths_tried_are_crossings_of_the_window() -> None:
    # A 5.75 m window at 80 m/s: 72 ms a crossing; 10, 20, 40 and 80 of them, the line's own 2 s
    # too, within 0.1 s and the shortest record (3 s here: 5.75 s left out).
    window = SimpleNamespace(
        acquisitions=[SimpleNamespace(receivers=[SimpleNamespace(x=0.0), SimpleNamespace(x=5.75)])]
    )
    profile = SimpleNamespace(records=[SimpleNamespace(duration_s=3.0)])

    lengths = segments.lengths_tried(2.0, [window], 80.0, SegmentRules(), profile)  # pyright: ignore[reportArgumentType]

    assert lengths == [0.72, 1.44, 2.0, 2.88]
