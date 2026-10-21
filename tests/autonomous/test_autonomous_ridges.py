"""Ridge tables: an image whose ridges are known, and active_test_profile's, whose
strongest ridge is the curve its picker followed."""

import numpy as np
import pytest
from autonomous_judged import JudgedLine
from sigpipe.base.acquisition import Acquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import VelocityType
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import load_image

from paco.autonomous.sensors.ridges import Maximum, format_ridge_table, ridge_table
from paco.qc.judging import saved_m0

FS = np.arange(0.0, 50.5, 0.5)  # Hz
VS = np.arange(1.0, 1001.0)  # m/s
# 24 receivers 0.25 m apart: a noise floor of 1/√24, an alias velocity of 0.5 m times f.
RECEIVERS = 24
SPACING = 0.25
# The ridges of the known image, at every frequency: velocity, strength (as the image keeps it,
# in float32), width.
STRONG = Maximum(velocity=300.0, strength=float(np.float32(0.9)), width=20.0, at_edge=False)
WEAK = Maximum(velocity=600.0, strength=float(np.float32(0.5)), width=10.0, at_edge=False)
EDGE = Maximum(velocity=1000.0, strength=float(np.float32(0.4)), width=0.0, at_edge=True)


def _known_image() -> DispersionImage:
    """At every frequency, over a background under the noise floor: a ridge 290-310 m/s strong,
    one 595-605 m/s weaker, a maximum at the grid's top velocity, and a bump under the floor at
    800 m/s. At 30 Hz, a spike at 5 m/s, slower than the alias velocity (15 m/s)."""
    fv = np.full((len(FS), len(VS)), 0.1)
    fv[:, (VS >= 290) & (VS <= 310)] = STRONG.strength
    fv[:, (VS >= 595) & (VS <= 605)] = WEAK.strength
    fv[:, -1] = EDGE.strength
    fv[:, VS == 800] = 0.15
    fv[np.flatnonzero(FS == 30.0)[0], np.flatnonzero(VS == 5.0)[0]] = 0.95
    origin = Coordinate(0.0, 0.0, 0.0)
    receivers = tuple(Coordinate(SPACING * index, 0.0, 0.0) for index in range(RECEIVERS))
    return DispersionImage(
        fv_map=fv,
        fs=FS,
        vs=VS,
        type=VelocityType.PHASE,
        acquisition=Acquisition(source=origin, receivers=receivers),
    )


def test_the_known_ridges_are_found_strongest_first() -> None:
    table = ridge_table(_known_image(), step_hz=5.0, band=(20.0, 30.0))

    assert (table.receivers, table.spacing_m) == (RECEIVERS, SPACING)
    assert table.floor == pytest.approx(1 / np.sqrt(RECEIVERS))
    assert [row.frequency for row in table.rows] == [20.0, 25.0, 30.0]
    assert [row.alias for row in table.rows] == [10.0, 12.5, 15.0]
    # The bump under the floor and the spike under the alias velocity are left out.
    assert all(row.maxima == (STRONG, WEAK, EDGE) for row in table.rows)


def test_a_row_keeps_the_strongest_maxima_and_rows_start_above_0_hz() -> None:
    image = _known_image()

    assert ridge_table(image, most=2, band=(30.0, 30.0)).rows[0].maxima == (STRONG, WEAK)
    rows = ridge_table(image).rows
    assert (len(rows), rows[0].frequency, rows[-1].frequency) == (20, 2.5, 50.0)


def test_the_table_as_the_model_reads_it() -> None:
    table = ridge_table(_known_image(), band=(30.0, 30.0))

    assert format_ridge_table(table).splitlines() == [
        "Ridges (24 receivers, 0.25 m apart; noise floor 0.20): each column's maxima above the "
        "floor and the alias velocity 2·dx·f, strongest first, as velocity m/s (strength, width "
        "m/s).",
        "f (Hz)  alias (m/s)  maxima",
        "  30.0           15  300 (0.90, 20) · 600 (0.50, 10) · 1000 (0.40, 0, at the grid's edge)",
    ]


@pytest.mark.slow
def test_the_strongest_ridge_is_the_picked_curve(judged_line: JudgedLine) -> None:
    folder = judged_line.folder / "xmid_89.25"
    curve = saved_m0(folder / CURVES_FILE)
    assert curve is not None
    order = np.argsort(curve.fs)

    # Within the curve's band, 16.7 to 30 Hz, where its ridge stands out: at 17 Hz a second ridge,
    # at 665 m/s, is nearly as strong.
    table = ridge_table(load_image(folder), band=(20.0, 30.0))

    assert len(table.rows) == 5
    for row in table.rows:
        picked = float(np.interp(row.frequency, curve.fs[order], curve.vs[order]))
        # Within 5 %: the picker follows the ridge with a smoothness of its own, and its curve is
        # kept at whole-metre wavelengths.
        assert row.maxima[0].velocity == pytest.approx(picked, rel=0.05), row.frequency
