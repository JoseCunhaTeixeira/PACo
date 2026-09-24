"""G6 on synthetic lines of models: Vs rising with depth along the xmids, with a jump the curves
do not show (non-uniqueness), one they show, a step in the ground, gaps and uneven depths."""

import numpy as np

from paco.inversion import InversionParameters
from paco.qc.g6_models import LINE, ModelProfileThresholds, judge_model_profile
from paco.qc.models import Flag, GateResult, Keep
from paco.qc.sides import Series

THRESHOLDS = ModelProfileThresholds()
DEPTHS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
PARAMETERS = InversionParameters(n_iterations=100_000)


def _model(xmid: int, scale: float = 1.0, deepest: float = 5.0) -> Series:
    depths = DEPTHS[deepest >= DEPTHS]
    return Series(f"xmid_{xmid:.2f}", float(xmid), depths, scale * (200.0 + 20.0 * depths))


def _curve(unit: str, *flags: str) -> GateResult:
    """G4's result on a window's curve, with the flags given (kept ones: G4 passed it)."""
    return GateResult(
        gate="G4",
        unit=unit,
        verdict="pass",
        flags=tuple(
            Flag(name=name, message="", stage="picking", action=Keep(note="")) for name in flags
        ),
    )


def _judge(
    models: list[Series],
    curve_flags: dict[str, tuple[str, ...]] | None = None,
    useful: dict[str, float | None] | None = None,
    without: tuple[float, ...] = (),
) -> dict[str, GateResult]:
    curve_flags = curve_flags or {}
    curves = {model.unit: _curve(model.unit, *curve_flags.get(model.unit, ())) for model in models}
    parameters = {model.unit: PARAMETERS for model in models}
    useful = useful if useful is not None else {model.unit: 5.0 for model in models}
    results = judge_model_profile(models, curves, parameters, useful, THRESHOLDS, without)
    return {result.unit: result for result in results}


def test_a_smooth_line_of_models_passes() -> None:
    results = _judge([_model(x) for x in range(8)])

    assert all(result.verdict == "pass" and result.flags == () for result in results.values())
    assert {metric.name: metric.value for metric in results[LINE].metrics} == {
        "models": 8,
        "without_model": 0,
        "useful_depth_spread": 0.0,
    }


def test_a_jump_the_curves_do_not_show_is_inverted_again() -> None:
    models = [_model(x, 1.3 if x == 4 else 1.0) for x in range(8)]
    results = _judge(models)

    jumped = results["xmid_4.00"]
    assert jumped.verdict == "retry"
    (flag,) = jumped.flags
    assert flag.name == "non_unique" and "on both sides" in flag.message
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "inversion",
        "overrides": {"n_iterations": 200_000, "n_burnin_iterations": 20_000},
    }
    assert all(results[f"xmid_{x}.00"].verdict == "pass" for x in (2, 3, 5, 6))


def test_a_jump_the_curves_show_too_is_kept() -> None:
    models = [_model(x, 1.3 if x == 4 else 1.0) for x in range(8)]
    results = _judge(models, {"xmid_4.00": ("shared_change",)})

    (flag,) = results["xmid_4.00"].flags
    assert flag.name == "model_change" and results["xmid_4.00"].verdict == "pass"
    assert flag.action.model_dump()["note"] == "the curves show the same change"
    # A curve without agreeing neighbours cannot tell either.
    results = _judge(models, {"xmid_4.00": ("no_neighbours",)})
    assert results["xmid_4.00"].flags[0].action.model_dump()["note"] == "the curves cannot tell"


def test_a_step_in_the_ground_is_kept() -> None:
    results = _judge([_model(x, 1.5 if x >= 4 else 1.0) for x in range(8)])

    assert all(result.verdict == "pass" for result in results.values())
    assert {flag.name for flag in results["xmid_3.00"].flags} == {"model_change"}
    assert {flag.name for flag in results["xmid_4.00"].flags} == {"model_change"}


def test_the_line_reports_gaps_and_uneven_useful_depths() -> None:
    models = [_model(x) for x in range(6)]
    useful: dict[str, float | None] = {m.unit: (5.0 if m.xmid < 3 else 1.5) for m in models}
    results = _judge(models, useful=useful, without=(6.0, 7.0))

    line = results[LINE]
    assert line.verdict == "pass"
    assert {flag.name for flag in line.flags} == {"gaps", "uneven_useful_depth"}
    assert "No model at xmid 6.00-7.00 (2)" in line.flags[0].message
    # A model informed down to its bottom counts at its deepest compared depth.
    results = _judge(models, useful={m.unit: None for m in models})
    assert results[LINE].flags == ()


def test_a_line_without_a_model_is_rejected() -> None:
    (line,) = judge_model_profile([], {}, {}, {}, THRESHOLDS, (1.0, 2.0))

    assert line.unit == LINE and line.verdict == "reject"
    assert {flag.name for flag in line.flags} == {"gaps"}
