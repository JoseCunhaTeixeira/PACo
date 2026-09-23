"""Measuring the quality of one dispersion image from its M0 pick."""

import numpy as np
from sigpipe.base import DispersionImage

from paco.picking import PickedMode
from paco.quality.models import Flag, ImageQuality, QualityParameters

# A point is on the data when its velocity is within this fraction of its column's brightest one.
_ON_DATA_TOLERANCE = 0.1
# Velocity grows like frequency when the local log-log slope of the pick is above this.
_CONSTANT_WAVELENGTH_SLOPE = 0.7
# The local slope is fitted over this many neighbouring points.
_SLOPE_POINTS = 7


def measure_quality(
    image: DispersionImage, m0: PickedMode | None, parameters: QualityParameters | None = None
) -> ImageQuality:
    """The quality of `image`, measured on the kept points of its M0 pick."""
    parameters = parameters or QualityParameters()
    if m0 is None:
        return ImageQuality(verdict="bad", flags=("no_ridge",), n_points=0, band_hz=None)

    # Frequency 0 has no wavelength: it cannot be measured.
    kept = m0.kept & (m0.frequencies > 0)
    frequencies, velocities = m0.frequencies[kept], m0.velocities[kept]
    if frequencies.size < 2:
        return ImageQuality(
            verdict="bad", flags=("no_ridge",), n_points=int(frequencies.size), band_hz=None
        )

    grid_v = image.vs.astype(float)
    rows = np.searchsorted(image.fs, frequencies)
    peaks = np.searchsorted(grid_v, velocities)
    above = np.clip(image.fv_map[rows].astype(float) - m0.noise_floor, 0, None)
    receivers = image.acquisition.receivers
    window_length = abs(receivers[-1].x - receivers[0].x)

    sharpness = _sharpness(above, peaks, grid_v, frequencies, velocities, window_length)
    prominence = _prominence(above, peaks)
    brightest = grid_v[np.argmax(above, axis=1)]
    on_data = float(np.mean(np.abs(brightest - velocities) / velocities < _ON_DATA_TOLERANCE))
    constant_wavelength = _constant_wavelength(frequencies, velocities)

    flags: list[Flag] = []
    if sharpness < parameters.min_sharpness:
        flags.append("sharpness")
    if prominence < parameters.min_prominence:
        flags.append("prominence")
    if on_data < parameters.min_on_data:
        flags.append("on_data")
    if constant_wavelength > parameters.max_constant_wavelength:
        flags.append("constant_wavelength")

    return ImageQuality(
        verdict="good" if not flags else "doubtful" if len(flags) == 1 else "bad",
        flags=tuple(flags),
        n_points=int(frequencies.size),
        band_hz=(float(frequencies.min()), float(frequencies.max())),
        sharpness=sharpness,
        prominence=prominence,
        on_data=on_data,
        constant_wavelength=constant_wavelength,
    )


def _sharpness(
    above: np.ndarray,
    peaks: np.ndarray,
    grid_v: np.ndarray,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    window_length: float,
) -> float:
    """Median peak width at half its height above the floor, over the resolution width."""
    widths = np.empty(peaks.size)
    for k, peak in enumerate(peaks):
        column, level = above[k], above[k, peak] / 2
        low = high = int(peak)
        while low > 0 and column[low - 1] >= level:
            low -= 1
        while high < column.size - 1 and column[high + 1] >= level:
            high += 1
        widths[k] = grid_v[high] - grid_v[low]
    resolution = velocities * (velocities / frequencies) / window_length
    return float(np.median(widths / resolution))


def _prominence(above: np.ndarray, peaks: np.ndarray) -> float:
    """Median peak height above the floor, over its column's median height above the floor."""
    heights = above[np.arange(peaks.size), peaks]
    background = np.maximum(np.median(above, axis=1), 1e-3)
    return float(np.median(heights / background))


def _constant_wavelength(frequencies: np.ndarray, velocities: np.ndarray) -> float:
    """Share of points where the local log-log slope of the pick is close to 1."""
    log_f, log_v = np.log(frequencies), np.log(velocities)
    half = _SLOPE_POINTS // 2
    slopes = np.array(
        [
            np.polyfit(
                log_f[max(0, k - half) : k + half + 1], log_v[max(0, k - half) : k + half + 1], 1
            )[0]
            for k in range(frequencies.size)
        ]
    )
    return float(np.mean(slopes > _CONSTANT_WAVELENGTH_SLOPE))
