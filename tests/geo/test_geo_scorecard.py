"""Scorecards (docs/geo_plan.md §5.3): active_p2's line, and cases with a known answer for coverage,
the longest gap, a variant against the best (PACo's rule 6) and dominance."""

import pytest
from geo_judged import JudgedLine

from paco.geo.sensors.scorecard import Scorecard, against, dominated, scorecard
from paco.qc.line_loop import LineRules


def test_the_lines_scorecard(judged_line: JudgedLine) -> None:
    card = scorecard(dict(judged_line.g3), judged_line.length_m, g4=judged_line.g4.values())

    # Two curves pass G3, at 17.25 and 89.25 m; the window between them fails.
    assert card.windows == (17.25, 53.25, 89.25, 125.25)
    assert card.covered == (17.25, 89.25)
    assert (card.coverage, card.longest_gap) == (0.5, (53.25,))
    # Half the longest wavelengths, 22 and 14 m; their 10th percentile, interpolated.
    assert card.depth_m == pytest.approx((11 + 7) / 2, abs=1e-5)
    assert card.depth_p10_m == pytest.approx(7 + 0.1 * (11 - 7), abs=1e-5)
    assert card.shallowest_m == pytest.approx((10 + 7) / 2, abs=1e-5)
    # Spans of 2.2 and 2.
    assert card.span == pytest.approx((2.2 + 2) / 2, abs=1e-5)
    assert card.precision == pytest.approx((0.098 + 0.063) / 2)
    assert (card.near_field, card.aliased, card.beyond) == (1.0, 0.0, 0.0)
    # G4: no outlier; misfits of 0.199 and 0.172.
    assert (card.outliers, card.misfit) == (0, pytest.approx((0.199 + 0.172) / 2))
    assert card.cpu_s is None


def test_coverage_counts_the_windows_that_count(judged_line: JudgedLine) -> None:
    # On the line, the windows accepted; a window whose processing failed is not covered.
    g3 = {**judged_line.g3, 161.25: None}

    card = scorecard(g3, judged_line.length_m, covered=(53.25, 89.25), cpu_s=12.5)

    assert card.coverage == pytest.approx(2 / 5)
    assert card.longest_gap == (125.25, 161.25)
    assert card.depth_m == pytest.approx((5 / 6 + 7) / 2, abs=1e-5)
    # Trial windows: no G4.
    assert (card.outliers, card.misfit, card.cpu_s) == (None, None, 12.5)


def test_a_variant_against_the_best_on_the_windows_both_judged() -> None:
    best = _card((1.0, 2.0))
    variant = _card((2.0, 3.0, 5.0), depth_m=8.5, windows=(1.0, 2.0, 3.0, 5.0))

    compared = against(variant, best)

    # Window 5 only the variant judged.
    assert (compared.gained, compared.lost, compared.shortened) == ((3.0,), (1.0,), False)


def test_a_variant_gaining_windows_by_shortening_its_curves_is_marked() -> None:
    # A variant may lose 20 % of the curves' longest wavelengths (PACo's QCConfig.line).
    best = _card((1.0,))

    assert against(_card((1.0, 2.0), depth_m=7.9), best).shortened
    assert not against(_card((1.0, 2.0), depth_m=8.1), best).shortened
    # Shorter, gaining nothing: nothing to mark.
    assert not against(_card((1.0,), depth_m=5.0), best).shortened
    # The run's own rules.
    assert against(
        _card((1.0, 2.0), depth_m=8.5), best, LineRules(max_wavelength_loss=0.1)
    ).shortened


def test_dominated_variants() -> None:
    cards = {
        "A": _card((1.0, 2.0, 3.0)),
        "B": _card((1.0, 2.0, 3.0), depth_m=9.0),  # A is as good, and deeper
        "C": _card((1.0, 2.0, 3.0, 4.0), depth_m=8.0, precision=0.3),  # it covers the most
        "D": _card((1.0, 2.0, 3.0)),  # as A: neither beats the other
        "E": _card((1.0, 2.0, 3.0), depth_m=None),  # its depth unknown, the worst
        "F": _card((1.0, 2.0, 3.0), precision=0.15),  # A, more precise
    }

    assert dominated(cards) == {"A": "F", "B": "A", "D": "F", "E": "A"}


def _card(
    covered: tuple[float, ...],
    depth_m: float | None = 10.0,
    precision: float | None = 0.2,
    windows: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
) -> Scorecard:
    """A scorecard of `windows` with only what coverage, depth and precision need."""
    return Scorecard(
        windows=windows,
        covered=covered,
        longest_gap=(),
        depth_m=depth_m,
        depth_p10_m=None,
        shallowest_m=None,
        span=None,
        precision=precision,
        near_field=None,
        beyond=None,
        aliased=None,
        outliers=None,
        misfit=None,
        cpu_s=None,
    )
