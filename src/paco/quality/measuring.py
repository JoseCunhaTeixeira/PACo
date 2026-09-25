"""Measuring the quality of one dispersion image from its M0 pick."""

import numpy as np
from sigpipe.base import DispersionImage

from paco.picking import PickedMode
from paco.picking.plane_waves import plane_wave_columns, prominences
from paco.quality.models import Flag, ImageQuality, QualityParameters

# A point is on the data when its velocity is within this fraction of its column's brightest one.
_ON_DATA_TOLERANCE = 0.1
# Velocity grows like frequency when the local log-log slope of the pick is above this.
_CONSTANT_WAVELENGTH_SLOPE = 0.7
# The local slope is fitted over this many neighbouring points.
_SLOPE_POINTS = 7
# Prominence is counted up to this: beyond, any ridge stands out enough. A long array's perfect
# plane wave has its sidelobes under the noise floor and would rise far above it, making any
# real image look weak; the old fixed limit, 2, is half of it.
_PROMINENT_ENOUGH = 4.0


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
    perfect = plane_wave_columns(image, frequencies, velocities, m0.noise_floor)
    sharpness = _sharpness(above, perfect, peaks, grid_v)
    prominence = _prominence(above, perfect, peaks)
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


def _half_width(column: np.ndarray, peak: int, grid_v: np.ndarray) -> float:
    """The width of the peak at `peak`, at half its height."""
    level = column[peak] / 2
    low = high = peak
    while low > 0 and column[low - 1] >= level:
        low -= 1
    while high < column.size - 1 and column[high + 1] >= level:
        high += 1
    return float(grid_v[high] - grid_v[low])


def _sharpness(
    above: np.ndarray, perfect: np.ndarray, peaks: np.ndarray, grid_v: np.ndarray
) -> float:
    """Median peak width at half its height above the floor, over a perfect plane wave's."""
    ratios = [
        _half_width(above[k], int(peak), grid_v) / width
        for k, peak in enumerate(peaks)
        if (width := _half_width(perfect[k], int(peak), grid_v)) > 0
    ]
    return float(np.median(ratios)) if ratios else 1.0


def _prominence(above: np.ndarray, perfect: np.ndarray, peaks: np.ndarray) -> float:
    """Median prominence (peak height above the floor over its column's median height above
    the floor, counted up to _PROMINENT_ENOUGH), over a perfect plane wave's."""
    measured = np.minimum(prominences(above, peaks), _PROMINENT_ENOUGH)
    reference = np.minimum(prominences(perfect, peaks), _PROMINENT_ENOUGH)
    return float(np.median(measured / np.maximum(reference, 1e-3)))


def _constant_wavelength(frequencies: np.ndarray, velocities: np.ndarray) -> float:
    """Share of points where the local log-log slope of the pick is close to 1."""
    return float(np.mean(local_slopes(frequencies, velocities) > _CONSTANT_WAVELENGTH_SLOPE))


def constant_wavelength_start(frequencies: np.ndarray, velocities: np.ndarray) -> float | None:
    """The shortest wavelength (m) of the stretch at the pick's long-wavelength end where
    velocity grows like frequency (the edge of what the window resolves): where to cut it.
    None when the longest wavelengths do not follow it."""
    wavelengths = velocities / frequencies
    along = local_slopes(frequencies, velocities) > _CONSTANT_WAVELENGTH_SLOPE
    start: float | None = None
    for index in np.argsort(wavelengths)[::-1]:
        if not along[index]:
            break
        start = float(wavelengths[index])
    return start


def local_slopes(frequencies: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """The pick's local log-log slope at each point, fitted over its neighbours."""
    log_f, log_v = np.log(frequencies), np.log(velocities)
    half = _SLOPE_POINTS // 2
    return np.array(
        [
            np.polyfit(
                log_f[max(0, k - half) : k + half + 1], log_v[max(0, k - half) : k + half + 1], 1
            )[0]
            for k in range(frequencies.size)
        ]
    )
