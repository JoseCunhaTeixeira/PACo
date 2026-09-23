"""Picking the modes of one dispersion image: M0, then each higher mode above the previous one."""

import math

import numpy as np

from paco.picking.models import PickedMode, PickingParameters
from paco.picking.tracking import corridor, lowest_ridge, track
from sigpipe.algorithms import lorentzian_uncertainty, min_resolvable_wavelength
from sigpipe.algorithms.picking.dispersion.curve import resample_wavelength
from sigpipe.base import DispersionCurve, DispersionImage, Mode


def pick_modes(
    image: DispersionImage, parameters: PickingParameters | None = None
) -> list[PickedMode]:
    """The modes of `image`, from M0 up; empty when not even M0 stands above the noise floor.

    Each mode is searched above the corridor of the one below it. The search stops at the first
    mode with fewer than `min_frequencies` kept points, or whose kept points have a median
    coherence under `mode_min_ratio` times the noise floor.
    """
    parameters = parameters or PickingParameters()
    frequencies = image.fs.astype(float)
    velocities = image.vs.astype(float)
    values = image.fv_map.astype(float)
    n_v = velocities.size

    # Phase-shift images are coherences: random phases sum to about 1/sqrt(N), not to 0.
    noise_floor = 1 / math.sqrt(len(image.acquisition.receivers))
    # Below twice the receiver spacing, wavelengths are aliased: the search starts above.
    shortest = min_resolvable_wavelength(image.acquisition) or 0.0
    start = np.searchsorted(velocities, shortest * frequencies, side="left")
    # Above the longest wavelength the window resolves, if limited, the search stops.
    stop = np.full(frequencies.size, n_v - 1)
    if parameters.max_wavelength is not None:
        receivers = image.acquisition.receivers
        longest = parameters.max_wavelength * abs(receivers[-1].x - receivers[0].x)
        stop = np.searchsorted(velocities, longest * frequencies, side="right") - 1

    modes: list[PickedMode] = []
    for number in range(parameters.max_modes):
        span = _longest_run(start < stop)
        if span is None or span.stop - span.start < parameters.min_frequencies:
            break

        ridge = lowest_ridge(values[span], start[span], stop[span], parameters.threshold)
        low, high = corridor(velocities, ridge, start[span], stop[span], parameters.corridor)
        path, on_edge = track(
            values[span], velocities, frequencies[span], low, high, parameters.smoothness
        )
        # A ridge cut by a search bound stays pinned, wherever the smoothing moved the pick.
        pinned = on_edge | (ridge == start[span]) | (ridge == stop[span])
        coherence = values[span][np.arange(path.size), path]
        ratio = coherence / noise_floor
        # Judged on its kept points only: pinned points are where a bound, not the data, decided,
        # and 0 Hz has no wavelength.
        kept = ~pinned & (ratio >= parameters.point_min_ratio) & (frequencies[span] > 0)
        # Points far below the mode's typical coherence are sidelobes or noise, not its ridge.
        if kept.any():
            kept &= coherence >= parameters.min_relative_coherence * np.median(coherence[kept])
        if kept.sum() < parameters.min_frequencies:
            break
        if float(np.median(ratio[kept])) < parameters.mode_min_ratio:
            break

        modes.append(
            PickedMode(
                number=number,
                frequencies=frequencies[span],
                velocities=velocities[path],
                coherence=coherence,
                pinned=pinned,
                kept=kept,
                noise_floor=noise_floor,
                curve=_curve(image, frequencies[span][kept], velocities[path][kept], number),
            )
        )

        # The next mode lies above this one's corridor, where this one was tracked.
        next_start = np.full_like(start, n_v)
        next_start[span] = high + 1
        start = next_start

    return modes


def _longest_run(mask: np.ndarray) -> slice | None:
    """The longest run of consecutive True values in `mask`, as a slice."""
    best: slice | None = None
    run_start: int | None = None
    for index, value in enumerate([*mask.tolist(), False]):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            if best is None or index - run_start > best.stop - best.start:
                best = slice(run_start, index)
            run_start = None
    return best


def _curve(
    image: DispersionImage, frequencies: np.ndarray, velocities: np.ndarray, number: int
) -> DispersionCurve | None:
    """The kept points as a sigpipe curve, like PAC's box picks: labelled M<n>, with Lorentzian
    uncertainties, resampled over wavelength."""
    if frequencies.size < 2:
        return None
    fs = frequencies.astype(np.float32)
    vs = velocities.astype(np.float32)
    curve = DispersionCurve(
        fs=fs,
        vs=vs,
        mode=Mode("M", number),
        acquisition=image.acquisition,
        vs_err=lorentzian_uncertainty(fs, vs, image.acquisition),
        type=image.type,
    )
    return resample_wavelength(curve)
