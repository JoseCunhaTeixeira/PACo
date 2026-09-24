"""The checks before S4: bounds derived from the curve, values given kept when they pass and
changed with a note when they do not."""

import numpy as np
import pytest
from sigpipe.base import DispersionCurve, Mode, VelocityType
from sigpipe.base.acquisition import UNKNOWN_ACQUISITION

from paco.inversion import InversionError
from paco.qc.priors import PriorRules, broadcast_layers, derive_inversion

RULES = PriorRules()
# 150 m/s at 30 m, rising to 300 m/s at 3 m: wavelengths 3 to 30 m, velocities 150 to 300 m/s.
WAVELENGTHS = np.array([30.0, 15.0, 9.0, 6.0, 3.0])
VELOCITIES = np.array([150.0, 180.0, 220.0, 260.0, 300.0])
CURVE = DispersionCurve(
    fs=VELOCITIES / WAVELENGTHS,
    vs=VELOCITIES,
    mode=Mode("M", 0),
    type=VelocityType.PHASE,
    acquisition=UNKNOWN_ACQUISITION,
)


def test_the_bounds_come_from_the_curve() -> None:
    derived = derive_inversion(CURVE, RULES)

    parameters = derived.parameters
    assert derived.notes == ()
    assert parameters.n_layers == 2
    # Vs from 0.8 x 150 to 1.5 x 300 m/s, the same for every layer, steps in PAC's proportion.
    for layer in parameters.vs_layers:
        assert (layer.vs_min, layer.vs_max) == (120.0, 450.0)
        assert layer.vs_perturb_std == pytest.approx(330 * 20 / 900, abs=0.05)
    # No layer thinner than a third of 3 m; the half-space no deeper than half of 30 m.
    (layer,) = parameters.thickness_layers
    assert (layer.thickness_min, layer.thickness_max) == (1.0, 15.0)
    # PAC's effort, with its burn-in.
    assert (parameters.n_iterations, parameters.n_burnin_iterations, parameters.n_chains) == (
        100_000,
        10_000,
        5,
    )


def test_more_layers_share_the_depth_the_curve_reaches() -> None:
    parameters = derive_inversion(CURVE, RULES, {"n_layers": 4}).parameters

    assert parameters.n_layers == 4
    assert [layer.thickness_max for layer in parameters.thickness_layers] == [5.0, 5.0, 5.0]


def test_values_given_are_kept_when_they_pass() -> None:
    given = {
        "vs_layers": [{"vs_min": 100.0, "vs_max": 800.0}] * 2,
        "thickness_layers": [{"thickness_min": 2.0, "thickness_max": 10.0}],
        "n_iterations": 20_000,
    }
    derived = derive_inversion(CURVE, RULES, given)

    assert derived.notes == ()
    parameters = derived.parameters
    assert [(layer.vs_min, layer.vs_max) for layer in parameters.vs_layers] == [(100.0, 800.0)] * 2
    assert parameters.thickness_layers[0].thickness_max == 10.0
    # The burn-in follows the iterations given.
    assert (parameters.n_iterations, parameters.n_burnin_iterations) == (20_000, 2_000)


def test_values_given_that_fail_are_changed_with_a_note() -> None:
    given = {
        "vs_layers": [{"vs_min": 200.0, "vs_max": 300.0}, {"vs_min": 100.0, "vs_max": 1000.0}],
        "thickness_layers": [{"thickness_min": 0.5, "thickness_max": 30.0}],
    }
    derived = derive_inversion(CURVE, RULES, given)

    assert derived.notes == (
        "vs_min above the curve's slowest velocity (150 m/s) in layer 1: set to 120 m/s.",
        "vs_max below 1.09 times the curve's fastest velocity (300 m/s) in layer 1: set to "
        "450 m/s.",
        "thickness_min thinner than the curve resolves (1 m) in layer 1: set to 1 m.",
        "thickness_max puts the half-space as deep as 30 m, below the 15.00 m the curve reaches: "
        "scaled by 0.50.",
    )
    first, second = derived.parameters.vs_layers
    assert (first.vs_min, first.vs_max) == (120.0, 450.0)
    assert (second.vs_min, second.vs_max) == (100.0, 1000.0)  # it brackets the curve
    (layer,) = derived.parameters.thickness_layers
    assert (layer.thickness_min, layer.thickness_max) == (1.0, 15.0)


def test_a_value_at_the_limit_within_rounding_passes() -> None:
    # The curve reaches 15 m: 15.0 given is the limit itself, not deeper.
    derived = derive_inversion(CURVE, RULES, {"thickness_layers": [{"thickness_max": 15.0}]})

    assert derived.notes == ()
    assert derived.reach_m == 15.0


def test_a_single_range_stands_for_every_layer() -> None:
    given = {"vs_layers": [{"vs_min": 100.0, "vs_max": 800.0}], "n_layers": 3}

    parameters = derive_inversion(CURVE, RULES, given).parameters

    assert [(layer.vs_min, layer.vs_max) for layer in parameters.vs_layers] == [(100.0, 800.0)] * 3
    assert broadcast_layers({"vs_layers": [{"vs_max": 300.0}]}) == {
        "vs_layers": [{"vs_max": 300.0}, {"vs_max": 300.0}]
    }
    # Two layers have one thickness: nothing to spread.
    assert broadcast_layers({"thickness_layers": [{"thickness_max": 4.0}]}) == {
        "thickness_layers": [{"thickness_max": 4.0}]
    }


def test_more_layers_than_the_curve_resolves_are_refused_with_a_note() -> None:
    derived = derive_inversion(CURVE, RULES, {"n_layers": 30})

    # 15 m of depth in layers of at least 1 m: 15 layers.
    assert derived.parameters.n_layers == 15
    assert derived.notes == (
        "n_layers 30: the curve resolves 15 (layers of at least 1.00 m down to 15.00 m); set to "
        "15.",
    )


def test_values_that_cannot_hold_are_an_error_for_the_agent() -> None:
    with pytest.raises(InversionError, match="Extra inputs are not permitted"):
        derive_inversion(CURVE, RULES, {"iterations": 5})
    with pytest.raises(InversionError, match="must exceed n_burnin_iterations"):
        derive_inversion(CURVE, RULES, {"n_iterations": 1_000, "n_burnin_iterations": 900})


def test_the_rules_are_configurable() -> None:
    rules = PriorRules(vs_low=0.5, vs_high=2.0, max_depth=1.0 / 3)
    parameters = derive_inversion(CURVE, rules).parameters

    assert (parameters.vs_layers[0].vs_min, parameters.vs_layers[0].vs_max) == (75.0, 600.0)
    assert parameters.thickness_layers[0].thickness_max == 10.0
    assert PriorRules.model_validate_json(rules.model_dump_json()) == rules
