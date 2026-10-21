"""Ridge tables: a dispersion image as text, for a model without vision. Every
few Hz, the strongest maxima of the image's column, each with its velocity, its strength and its
width; with the noise floor random phases sum to (1/√N, sigpipe's) and the alias velocity 2·dx·f,
below which a maximum may be the alias of another (G2's limit)."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from sigpipe.algorithms.picking.dispersion.curve import min_resolvable_wavelength
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.quality.image import noise_floor

# The rows' frequency step (Hz), and the maxima a row keeps.
STEP_HZ = 2.5
MOST = 3


@dataclass(frozen=True, slots=True)
class Maximum:
    """A maximum of an image's column."""

    velocity: float  # m/s
    strength: float  # the image's value there: 1 for a perfect plane wave
    width: float  # m/s, where the column stands above half the maximum's height over the floor
    at_edge: bool  # on the velocity grid's first or last value: the grid may cut the ridge


@dataclass(frozen=True, slots=True)
class Row:
    """An image's column: its maxima above the noise floor and the alias velocity, strongest
    first."""

    frequency: float  # Hz
    alias: float  # m/s, 2·dx·f
    maxima: tuple[Maximum, ...]


@dataclass(frozen=True, slots=True)
class RidgeTable:
    """An image's ridges, a row every few Hz."""

    receivers: int
    spacing_m: float | None  # the smallest receiver spacing; None when the geometry is unknown
    floor: float  # what random phases sum to
    rows: tuple[Row, ...]


def ridge_table(
    image: DispersionImage,
    step_hz: float = STEP_HZ,
    most: int = MOST,
    band: tuple[float, float] | None = None,
) -> RidgeTable:
    """`image`'s ridge table: its columns nearest each multiple of `step_hz` within `band` (by
    default the image's frequencies above 0), each with its `most` strongest maxima."""
    floor = noise_floor(image)
    shortest = min_resolvable_wavelength(image.acquisition)
    fs = image.fs
    low, high = band if band is not None else (float(fs[0]), float(fs[-1]))
    first = max(math.ceil(low / step_hz), 1)
    columns: list[int] = []
    for step in range(first, math.floor(high / step_hz) + 1):
        column = int(np.argmin(np.abs(fs - step * step_hz)))
        if column not in columns:
            columns.append(column)
    rows = tuple(_row(image, column, floor, shortest, most) for column in columns)
    return RidgeTable(
        receivers=len(image.acquisition.receivers),
        spacing_m=shortest / 2 if shortest else None,
        floor=floor,
        rows=rows,
    )


def format_ridge_table(table: RidgeTable) -> str:
    """`table` as the model reads it: what it shows, then a row a frequency."""
    spacing = f"{table.spacing_m:g} m apart" if table.spacing_m else "spacing unknown"
    lines = [
        f"Ridges ({table.receivers} receivers, {spacing}; noise floor {table.floor:.2f}): each "
        "column's maxima above the floor and the alias velocity 2·dx·f, strongest first, as "
        "velocity m/s (strength, width m/s).",
        "f (Hz)  alias (m/s)  maxima",
    ]
    for row in table.rows:
        maxima = " · ".join(_described(maximum) for maximum in row.maxima) or "none"
        lines.append(f"{row.frequency:6.1f}  {row.alias:11.0f}  {maxima}")
    return "\n".join(lines)


def _row(
    image: DispersionImage, column: int, floor: float, shortest: float | None, most: int
) -> Row:
    """The row of `image`'s column `column`; no alias velocity when the geometry is unknown."""
    frequency = float(image.fs[column])
    alias = shortest * frequency if shortest else 0.0
    return Row(frequency, alias, _maxima(image.fv_map[column], image.vs, floor, alias, most))


def _maxima(
    column: NDArray[np.floating], vs: NDArray[np.floating], floor: float, alias: float, most: int
) -> tuple[Maximum, ...]:
    """`column`'s `most` strongest maxima above `floor`, at velocities from `alias` up; one at the
    grid's edge counts."""
    padded = np.concatenate(([column.min() - 1.0], column, [column.min() - 1.0]))
    peaks, _ = find_peaks(padded)
    found = [
        Maximum(
            velocity=float(vs[index]),
            strength=float(column[index]),
            width=_width(column, vs, index, floor),
            at_edge=index in (0, len(column) - 1),
        )
        for index in (int(peak) - 1 for peak in peaks)
        if column[index] > floor and vs[index] >= alias
    ]
    return tuple(sorted(found, key=lambda maximum: maximum.strength, reverse=True)[:most])


def _width(
    column: NDArray[np.floating], vs: NDArray[np.floating], index: int, floor: float
) -> float:
    """The velocities around maximum `index` where `column` stands above half its height over
    `floor`, on the grid."""
    level = floor + (column[index] - floor) / 2
    left = right = index
    while left > 0 and column[left - 1] >= level:
        left -= 1
    while right < len(column) - 1 and column[right + 1] >= level:
        right += 1
    return float(vs[right] - vs[left])


def _described(maximum: Maximum) -> str:
    edge = ", at the grid's edge" if maximum.at_edge else ""
    return f"{maximum.velocity:.0f} ({maximum.strength:.2f}, {maximum.width:.0f}{edge})"
