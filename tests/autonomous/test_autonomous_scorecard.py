"""Scorecards: active_test_profile's line, and cases with a known answer for coverage,
the longest gap, a variant against the best and dominance."""

import pytest
from autonomous_judged import JudgedLine

from paco.autonomous.sensors.scorecard import Scorecard, against, dominated, scorecard
from paco.qc.models import GateResult, Metric


@pytest.mark.slow
def test_the_lines_scorecard(judged_line: JudgedLine) -> None:
    card = scorecard(dict(judged_line.g3), judged_line.length_m, g4=judged_line.g4.values())

    # Three curves pass G3, at 17.25, 89.25 and 125.25 m; the window at 53.25 m fails.
    assert card.windows == (17.25, 53.25, 89.25, 125.25)
    assert card.covered == (17.25, 89.25, 125.25)
    assert (card.coverage, card.longest_gap) == (0.75, (53.25,))
    # Half the longest wavelengths, 15, 16 and 13 m; their 10th percentile, interpolated.
    assert card.depth_m == pytest.approx(7.5, abs=1e-5)
    assert card.depth_p10_m == pytest.approx(6.5 + 0.2 * (7.5 - 6.5), abs=1e-5)
    # The shortest wavelengths, 11, 8 and 9 m.
    assert card.shallowest_m == pytest.approx(9, abs=1e-5)
    # Spans of 1.36, 2 and 1.44.
    assert card.span == pytest.approx(13 / 9, abs=1e-5)
    assert card.precision == pytest.approx(0.073)
    assert (card.near_field, card.aliased, card.beyond) == (1.0, 0.0, 0.0)
    # G4: one outlier, 17.25's; misfits of 0.231, 0.212 and 0.095.
    assert (card.outliers, card.misfit) == (1, pytest.approx(0.212))
    assert card.cpu_s is None


@pytest.mark.slow
def test_coverage_counts_the_windows_that_count(judged_line: JudgedLine) -> None:
    # On the line, the windows accepted; a window whose processing failed is not covered.
    g3 = {**judged_line.g3, 161.25: None}

    card = scorecard(g3, judged_line.length_m, covered=(53.25, 89.25), cpu_s=12.5)

    assert card.coverage == pytest.approx(2 / 5)
    assert card.longest_gap == (125.25, 161.25)
    assert card.depth_m == pytest.approx((1.72603 / 2 + 8) / 2, abs=1e-5)
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
    assert against(_card((1.0, 2.0), depth_m=8.5), best, loss=0.1).shortened


def test_the_images_are_scored_by_their_coherence_their_background_and_the_picks_on_them() -> None:
    # A clean image and a striped one, as 11 and 20 receivers gave across a lateral change.
    g2 = {
        1.0: _result(
            "G2", coherent_columns=1.0, lateral_coherence=0.96, background=0.1, stripes=0.002
        ),
        2.0: _result(
            "G2", coherent_columns=0.71, lateral_coherence=0.78, background=0.3, stripes=0.02
        ),
    }
    g3 = {1.0: _result("G3", on_data=0.89), 2.0: _result("G3", on_data=0.8)}

    card = scorecard(g3, 15.0, g2=g2)

    assert card.coherent == pytest.approx(0.855)
    assert card.lateral == pytest.approx(0.87)
    assert card.on_ridge == pytest.approx(0.845)
    assert (card.background, card.stripes) == (pytest.approx(0.2), pytest.approx(0.011))
    # Each window's mean, its background counted as 1 minus it; their median, and the 10th
    # percentile for the weakest windows.
    clean, striped = (1.0 + 0.96 + 0.89 + 0.9) / 4, (0.71 + 0.78 + 0.8 + 0.7) / 4
    assert card.image == pytest.approx((clean + striped) / 2)
    assert card.image_p10 == pytest.approx(striped + 0.1 * (clean - striped))
    # Without G2's results, the picks alone.
    assert scorecard(g3, 15.0).image == pytest.approx((0.89 + 0.8) / 2)
    assert scorecard({1.0: None}, 15.0).image is None


def test_dominated_variants() -> None:
    # The images first, then the coverage and the depth; the picks' uncertainty, the geometry's,
    # does not count.
    cards = {
        "A": _card((1.0, 2.0, 3.0)),
        "B": _card((1.0, 2.0, 3.0), depth_m=9.0),  # A is as good, and deeper
        "C": _card((1.0, 2.0, 3.0, 4.0), depth_m=8.0, image=0.8),  # it covers the most
        "D": _card((1.0, 2.0, 3.0)),  # as A: neither beats the other
        "E": _card((1.0, 2.0, 3.0), depth_m=None),  # its depth unknown, the worst
        "F": _card((1.0, 2.0, 3.0), image=0.95),  # A, with cleaner images
        "G": _card((1.0, 2.0, 3.0), precision=0.1),  # A, more precise: no better
    }

    assert dominated(cards) == {"A": "F", "B": "A", "D": "F", "E": "A", "G": "F"}


def _result(gate: str, **values: float) -> GateResult:
    """A gate's passing result on a window, with `values` its measures."""
    measured = tuple(Metric(name=name, value=value, passed=True) for name, value in values.items())
    return GateResult(gate=gate, unit="xmid_1.00", verdict="pass", metrics=measured)


def _card(
    covered: tuple[float, ...],
    depth_m: float | None = 10.0,
    precision: float | None = 0.2,
    windows: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    image: float | None = 0.9,
) -> Scorecard:
    """A scorecard of `windows` with only what coverage, depth, precision and images need."""
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
        image=image,
    )
